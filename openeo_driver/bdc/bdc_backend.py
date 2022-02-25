import numbers
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Any
from unittest.mock import Mock
from collections import OrderedDict
from datacube.model import DatasetType

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection

from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo.metadata import CollectionMetadata, Band
from openeo_driver.ProcessGraphDeserializer import ConcreteProcessing, SimpleProcessing
from openeo_driver.backend import (SecondaryServices, OpenEoBackendImplementation, CollectionCatalog, ServiceMetadata,
                                   BatchJobs, BatchJobMetadata, OidcProvider, UserDefinedProcesses,
                                   UserDefinedProcessMetadata, LoadParameters)
from openeo_driver.datacube import DriverDataCube
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.errors import JobNotFoundException, JobNotFinishedException, ProcessGraphNotFoundException
from openeo_driver.save_result import AggregatePolygonResult
from openeo_driver.users import User
from openeo_driver.utils import EvalEnv

# from .datacube-explorer.cubedash import generate
import datacube.index
import datacube.config

import json

DEFAULT_DATETIME = datetime(2020, 4, 23, 16, 20, 27)

# TODO: eliminate this global state with proper pytest fixture usage!
_collections = {}
_load_collection_calls = {}


def datacube_index(datacube_config_path=None) -> Union['datacube.index.index.Index', Any]:
    """Retrieve a ODC Index connection
    Args:
        datacube_config_path (str): Path to datacube's database connection config
    Returns:
        None or datacube.index.index.Index object
    """

    datacube_config = datacube.config.LocalConfig.find(datacube_config_path)
    return datacube.index.index_connect(datacube_config, 'bdc_openeo_backend')


def odc_collection2stac_metadata(collection: DatasetType):
    dc = datacube.api.Datacube()
    def get_bounds(datasets, crs) -> datacube.utils.geometry._base.Geometry:
        from datacube.utils import geometry
        bbox = geometry.bbox_union(ds.extent.to_crs(crs).boundingbox for ds in datasets)
        return geometry.box(*bbox, crs=crs)

    ds = dc.find_datasets(product=collection.name)
    crs = collection.definition.get('storage', {}).get('crs', 'EPSG:4326')
    bbox = get_bounds(ds, crs).boundingbox
    times = sorted([t.center_time for t in ds])
    interval = []
    if len(times):
        interval.append(times[0].strftime("%Y-%m-%d"))
        if len(times) > 1:
            interval.append(times[-1].strftime("%Y-%m-%d"))
        else:
            interval.append(None)

    definition = collection.definition
    col = OrderedDict()
    col['id'] = collection.name
    col['product_id'] = collection.id
    col['title'] = collection.name
    col['name'] = collection.name
    col['description'] = definition["description"]
    col['license'] = 'free'
    col["extent"] = {
        "spatial": {
            "bbox": [
                [bbox.left, bbox.bottom, bbox.right, bbox.top]
            ]
        },
        "temporal": {
            "interval": [interval,]
        }
    }
    col['cube:dimensions'] = {
        "x": {"type": "spatial", "extent": [bbox.left, bbox.right], "step": 10, "reference_system": crs},
        "y": {"type": "spatial", "extent": [bbox.bottom, bbox.top], "step": 10, "reference_system": crs},
        "t": {"type": "temporal", "extent": [interval,]},
        "bands": {"type": "bands", "values": [k for k in collection.measurements]},
    }
    col['summaries'] = {
                "eo:bands": [
                    {
                        "name": k, 
                        "common_name": collection.measurements[k]['aliases'][-1] if 'aliases' in collection.measurements[k] else k
                    } for k in collection.measurements]
            },
    col['links'] = []
    return col


def utcnow() -> datetime:
    # To simplify testing, we break time.
    return DEFAULT_DATETIME


def get_collection(collection_id: str) -> 'BDCDriverDataCube':
    return _collections[collection_id]


def _register_load_collection_call(collection_id: str, load_params: LoadParameters):
    if collection_id not in _load_collection_calls:
        _load_collection_calls[collection_id] = []
    _load_collection_calls[collection_id].append(load_params.copy())


def all_load_collection_calls(collection_id: str) -> List[LoadParameters]:
    return _load_collection_calls[collection_id]


def last_load_collection_call(collection_id: str) -> LoadParameters:
    return _load_collection_calls[collection_id][-1]


def reset():
    # TODO: can we eliminate reset now?
    global _collections, _load_collection_calls
    _collections = {}
    _load_collection_calls = {}


class BDCProcessGraphVisitor(ProcessGraphVisitor):

    def __init__(self):
        super(BDCProcessGraphVisitor, self).__init__()
        self.processes = []

    def enterProcess(self, process_id: str, arguments: dict, namespace: Union[str, None]):
        self.processes.append((process_id, arguments, namespace))

    def constantArgument(self, argument_id: str, value):
        if isinstance(value, numbers.Real):
            pass
        elif isinstance(value, str):
            pass
        else:
            raise ValueError(
                'Only numeric constants are accepted, but got: ' + str(value) + ' for argument: ' + str(
                    argument_id))
        return self


class BDCSecondaryServices(SecondaryServices):
    _registry = [
        ServiceMetadata(
            id="bdc-wms",
            process={},
            url='http://brazildatacube.dpi.inpe.br/bdc/geoserver/bdc_catalog/wms',
            type="WMS",
            enabled=True,
            configuration={"version": "1.1.1"},
            attributes={},
            title="Brazil Data Cube WMS Service",
            created=datetime(2020, 4, 9, 15, 5, 8)
        )
    ]

    def _create_service(self, user_id: str, process_graph: dict, service_type: str, api_version: str,
                        configuration: dict) -> str:
        service_id = 'c63d6c27-c4c2-4160-b7bd-9e32f582daec'
        return service_id

    def service_types(self) -> dict:
        return {
            "WMTS": {
                "title": "Web Map Tile Service",
                "configuration": {
                    "version": {
                        "type": "string",
                        "description": "The WMTS version to use.",
                        "default": "1.0.0",
                        "enum": [
                            "1.0.0"
                        ]
                    }
                },
                "process_parameters": [
                    # TODO: we should at least have bbox and time range parameters here
                ],
                "links": [],
            }
        }

    def list_services(self, user_id: str) -> List[ServiceMetadata]:
        return self._registry

    def service_info(self, user_id: str, service_id: str) -> ServiceMetadata:
        return next(s for s in self._registry if s.id == service_id)

    def get_log_entries(self, service_id: str, user_id: str, offset: str) -> List[dict]:
        return [
            {"id": 3, "level": "info", "message": "Loaded data."}
        ]


def mock_side_effect(fun):
    """
    Decorator to flag a BDCDataCube method to be wrapped in Mock(side_effect=...)
    so that it allows to mock-style inspection (call_count, assert_called_once, ...)
    while still providing a real implementation.
    """
    fun._mock_side_effect = True
    return fun


class BDCConfig(dict):
    def __init__(self, config_file="./openeo_driver/bdc/config.json"):
        with open(config_file) as json_file:
            data = json.load(json_file)
        super(BDCConfig, self).__init__(data)
    
    @property
    def secondary_services(self):
        return self.get('secondary_services', {})

    @property
    def catalog(self):
        return self.get('catalog', {})

    @property
    def batch_jobs(self):
        return self.get('batch_jobs', {})

    @property
    def user_defined_processes(self):
        return self.get('user_defined_processes', {})

    @property
    def processing(self):
        return self.get('processing', {})

class BDCDriverDataCube(DriverDataCube):

    def __init__(self, metadata: CollectionMetadata = None):
        super(BDCDriverDataCube, self).__init__(metadata=metadata)

        # TODO #47: remove this non-standard process?
        self.timeseries = Mock(name="timeseries", return_value={})

        # TODO can we get rid of these non-standard "apply_tiles" processes?
        self.apply_tiles = Mock(name="apply_tiles", return_value=self)
        self.apply_tiles_spatiotemporal = Mock(
            name="apply_tiles_spatiotemporal", return_value=self)

        # Create mock methods for remaining data cube methods that are not yet defined
        already_defined = set(BDCDriverDataCube.__dict__.keys()
                              ).union(self.__dict__.keys())
        for name, method in DriverDataCube.__dict__.items():
            if not name.startswith('_') and name not in already_defined and callable(method):
                setattr(self, name, Mock(name=name, return_value=self))

        for name in [n for n, m in BDCDriverDataCube.__dict__.items() if getattr(m, '_mock_side_effect', False)]:
            setattr(self, name, Mock(side_effect=getattr(self, name)))

    @mock_side_effect
    def reduce_dimension(self, reducer, dimension: str, env: EvalEnv) -> 'BDCDriverDataCube':
        self.metadata = self.metadata.reduce_dimension(
            dimension_name=dimension)
        return self

    @mock_side_effect
    def add_dimension(self, name: str, label, type: str = "other") -> 'BDCDriverDataCube':
        self.metadata = self.metadata.add_dimension(
            name=name, label=label, type=type)
        return self

    def save_result(self, filename: str, format: str, format_options: dict = None) -> str:
        with open(filename, "w") as f:
            f.write("{f}:save_result({s!r}".format(f=format, s=self))
        return filename

    def zonal_statistics(self, regions, func, scale=1000, interval="day") -> 'AggregatePolygonResult':
        # TODO: get rid of non-standard "zonal_statistics" (standard process is "aggregate_spatial")
        assert func == 'mean' or func == 'avg'

        def assert_polygon_or_multipolygon(geometry):
            assert isinstance(geometry, Polygon) or isinstance(
                geometry, MultiPolygon)

        if isinstance(regions, str):
            geometries = [
                geometry for geometry in DelayedVector(regions).geometries]

            assert len(geometries) > 0
            for geometry in geometries:
                assert_polygon_or_multipolygon(geometry)
        elif isinstance(regions, GeometryCollection):
            assert len(regions) > 0
            for geometry in regions:
                assert_polygon_or_multipolygon(geometry)
        else:
            assert_polygon_or_multipolygon(regions)

        return AggregatePolygonResult(timeseries={
            "2015-07-06T00:00:00": [2.345],
            "2015-08-22T00:00:00": [float('nan')]
        }, regions=GeometryCollection())


class BDCCollectionCatalog(CollectionCatalog):
    def __init__(self, config: BDCConfig):
        self.dc_index = datacube_index()
        self._COLLECTIONS = [
            odc_collection2stac_metadata(c) 
            for c in self.dc_index.products.get_all() 
            if c.name not in config.catalog["disabled_collections"]]
        
        super().__init__(all_metadata=self._COLLECTIONS)

    def load_collection(self, collection_id: str, load_params: LoadParameters, env: EvalEnv) -> BDCDriverDataCube:
        _register_load_collection_call(collection_id, load_params)
        if collection_id in _collections:
            return _collections[collection_id]

        image_collection = BDCDriverDataCube(
            metadata=CollectionMetadata(
                metadata=self.get_collection_metadata(collection_id))
        )

        _collections[collection_id] = image_collection
        return image_collection


class BDCBatchJobs(BatchJobs):
    _job_registry = {}

    def generate_job_id(self):
        return str(uuid.uuid4())

    def create_job(
            self, user_id: str, process: dict, api_version: str,
            metadata: dict, job_options: dict = None
    ) -> BatchJobMetadata:
        job_id = self.generate_job_id()
        job_info = BatchJobMetadata(
            id=job_id, status="created", process=process, created=utcnow(), job_options=job_options,
            title=metadata.get("title"), description=metadata.get("description")
        )
        self._job_registry[(user_id, job_id)] = job_info
        return job_info

    def get_job_info(self, job_id: str, user: User) -> BatchJobMetadata:
        return self._get_job_info(job_id=job_id, user_id=user.user_id)

    def _get_job_info(self, job_id: str, user_id: str) -> BatchJobMetadata:
        try:
            return self._job_registry[(user_id, job_id)]
        except KeyError:
            raise JobNotFoundException(job_id)

    def get_user_jobs(self, user_id: str) -> List[BatchJobMetadata]:
        return [v for (k, v) in self._job_registry.items() if k[0] == user_id]

    @classmethod
    def _update_status(cls, job_id: str, user_id: str, status: str):
        try:
            cls._job_registry[(user_id, job_id)] = cls._job_registry[(
                user_id, job_id)]._replace(status=status)
        except KeyError:
            raise JobNotFoundException(job_id)

    def start_job(self, job_id: str, user: User):
        self._update_status(
            job_id=job_id, user_id=user.user_id, status="running")

    def _output_root(self) -> Path:
        return Path("/data/jobs")

    def get_results(self, job_id: str, user_id: str) -> Dict[str, dict]:
        if self._get_job_info(job_id=job_id, user_id=user_id).status != "finished":
            raise JobNotFinishedException
        return {
            "output.tiff": {
                "output_dir": str(self._output_root() / job_id),
                "media_type": "image/tiff; application=geotiff",
                "bands": [Band(name="NDVI", common_name="NDVI", wavelength_um=1.23)],
                "nodata": 123,
                "instruments": "MSI"
            }
        }

    def get_log_entries(self, job_id: str, user_id: str, offset: Optional[str] = None) -> List[dict]:
        self._get_job_info(job_id=job_id, user_id=user_id)
        return [
            {"id": "1", "level": "info", "message": "hello world"}
        ]

    def cancel_job(self, job_id: str, user_id: str):
        self._get_job_info(job_id=job_id, user_id=user_id)

    def delete_job(self, job_id: str, user_id: str):
        self.cancel_job(job_id, user_id)


class BDCUserDefinedProcesses(UserDefinedProcesses):
    def __init__(self):
        super().__init__()
        self._processes: Dict[Tuple[str, str], UserDefinedProcessMetadata] = {}

    def reset(self, db: Dict[Tuple[str, str], UserDefinedProcessMetadata]):
        self._processes = db

    def get(self, user_id: str, process_id: str) -> Union[UserDefinedProcessMetadata, None]:
        return self._processes.get((user_id, process_id))

    def get_for_user(self, user_id: str) -> List[UserDefinedProcessMetadata]:
        return [udp for key, udp in self._processes.items() if key[0] == user_id]

    def save(self, user_id: str, process_id: str, spec: dict) -> None:
        self._processes[user_id,
                        process_id] = UserDefinedProcessMetadata.from_dict(spec)

    def delete(self, user_id: str, process_id: str) -> None:
        try:
            self._processes.pop((user_id, process_id))
        except KeyError:
            raise ProcessGraphNotFoundException(process_id)



class BDCBackendImplementation(OpenEoBackendImplementation):
    def __init__(self):
        from BDCConcreteProcessing import BDCConcreteProcessing
        bdc_config = BDCConfig()
        super(BDCBackendImplementation, self).__init__(
            secondary_services=BDCSecondaryServices(),
            catalog=BDCCollectionCatalog(bdc_config),
            batch_jobs=BDCBatchJobs(),
            user_defined_processes=BDCUserDefinedProcesses(),
            # processing=SimpleProcessing(),
            processing=BDCConcreteProcessing(),
            # processing=ConcreteProcessing(),
        )

    def oidc_providers(self) -> List[OidcProvider]:
        return [
            OidcProvider(id="testprovider", issuer="https://oidc.oeo.net",
                         scopes=["openid"], title="Test"),
            OidcProvider(
                id="eoidc", issuer="https://eo.id", scopes=["openid"], title="e-OIDC",
                default_client={"id": "badcafef00d"},
                default_clients=[{
                    "id": "badcafef00d",
                    "grant_types": ["urn:ietf:params:oauth:grant-type:device_code+pkce", "refresh_token"]
                }],
            ),
            # Allow testing with Keycloak setup running in docker on localhost.
            OidcProvider(
                id="local", title="Local Keycloak",
                issuer="http://localhost:9090/auth/realms/master", scopes=["openid"],
            ),
        ]

    def file_formats(self) -> dict:
        return {
            "input": {
                "GeoJSON": {
                    "gis_data_types": ["vector"],
                    "parameters": {},
                }
            },
            "output": {
                "GTiff": {
                    "title": "GeoTiff",
                    "gis_data_types": ["raster"],
                    "parameters": {},
                },
                # "STAC": {
                #     "title": "STAC",
                #     "gis_data_types": ["raster"],
                #     "parameters": {},
                # },
                # "WMS": {
                #     "title": "WMS",
                #     "gis_data_types": ["raster"],
                #     "parameters": {},
                # },
            },
        }

    def load_disk_data(
            self, format: str, glob_pattern: str, options: dict, load_params: LoadParameters, env: EvalEnv
    ) -> BDCDriverDataCube:
        _register_load_collection_call(glob_pattern, load_params)
        return BDCDriverDataCube()

    def visit_process_graph(self, process_graph: dict) -> ProcessGraphVisitor:
        return BDCProcessGraphVisitor().accept_process_graph(process_graph)
