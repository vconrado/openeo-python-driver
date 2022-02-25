from openeo_driver.ProcessGraphDeserializer import SimpleProcessing, _add_standard_processes, _OPENEO_PROCESSES_PYTHON_WHITELIST, ConcreteProcessing, SPECS_ROOT, load_collection, save_result
from openeo_driver.processes import ProcessRegistry
from openeo_driver.ProcessGraphDeserializer import evaluate, ProcessSpec, ProcessFunction
from pathlib import Path
from openeo.capabilities import ComparableVersion
from typing import Union, Callable, List, Dict
from openeo_driver.utils import EvalEnv
from bdc_backend import BDCBackendImplementation

_BDC_PROCESSES_PYTHON_WHITELIST = []

bdc_registry = ProcessRegistry(spec_root=SPECS_ROOT / 'openeo-processes/1.x', argument_names=["args", "env"])
_add_standard_processes(bdc_registry, _BDC_PROCESSES_PYTHON_WHITELIST)

def bdc_non_standard_process(spec: ProcessSpec, registries: List[ProcessRegistry]) -> Callable[[ProcessFunction], ProcessFunction]:
    """Decorator for registering non-standard process functions"""

    def decorator(f: ProcessFunction) -> ProcessFunction:
        for registry in registries:
            registry.add_function(f=f, spec=spec.to_dict_100())
        return f

    return decorator

# Permirte arrastar a collection no web editor
bdc_registry.add_function(load_collection)
bdc_registry.add_function(save_result)

@bdc_non_standard_process(
    # TODO: get spec directly from @process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/atmospheric_correction.json"))
    ProcessSpec(
        id='bdc_ard',
        description="Create ARD data from raw data",
        extra={
            "summary": "Apply atmospheric correction and create cloud mask",
            "categories": ["cubes", "optical"],
            "experimental": True,
            "links": [
                {
                    "rel": "about",
                    "href": "http://brazildatacube.dpi.inpe.br/openeo/info/bdc_ard",
                    "title": "Apply atmospheric correction and create cloud mask"
                }
            ],
            "exceptions": {
                "DigitalElevationModelInvalid": {
                    "message": "The digital elevation model specified is either not a DEM or can't be used with the data cube given."
                }
            },
        }
    )
        .param('data', description="Data cube containing multi-spectral optical top of atmosphere reflectances to be corrected.", schema={"type": "object", "subtype": "raster-cube"})
        .returns(description="the corrected data as a data cube", schema={"type": "object", "subtype": "raster-cube"}),
    registries=[bdc_registry,]
)
def bdc_ard(args: Dict, env: EvalEnv) -> object:
    pass

@bdc_non_standard_process(
    ProcessSpec(
        id='bdc_crop_scene',
        description="Crop scene",
        extra={
            "summary": "Crop scene by region",
            "categories": ["cubes", "optical"],
            "experimental": True,
            "links": [
                {
                    "rel": "about",
                    "href": "http://brazildatacube.dpi.inpe.br/openeo/info/bdc_crop_scene",
                    "title": "Crop scene by region"
                }
            ],
            "exceptions": {
                "DigitalElevationModelInvalid": {
                    "message": "The digital elevation model specified is either not a DEM or can't be used with the data cube given."
                }
            },
        }
    )
        .param('data', description="Data cube containing multi-spectral optical top of atmosphere reflectances to be corrected.", schema={"type": "object", "subtype": "raster-cube"})
        .param('region', description='Region',schema={"type": "string", "enum": ["curuai", "baixo_amazonas"]})
        .returns(description="the corrected data as a data cube", schema={"type": "object", "subtype": "raster-cube"}),
    registries=[bdc_registry,]
)
def bdc_crop_scene(args: Dict, env: EvalEnv) -> object:
    pass


@bdc_non_standard_process(
    ProcessSpec(
        id='bdc_publish_stac',
        description="Crop scene",
        extra={
            "summary": "Publish STAC",
            "categories": ["cubes", "optical"],
            "experimental": True,
            "links": [
                {
                    "rel": "about",
                    "href": "http://brazildatacube.dpi.inpe.br/openeo/info/bdc_publish_stac",
                    "title": "Publish STAC"
                }
            ],
            "exceptions": {
                "DigitalElevationModelInvalid": {
                    "message": "The digital elevation model specified is either not a DEM or can't be used with the data cube given."
                }
            },
        }
    )
        .param('data', description="Data cube containing multi-spectral optical top of atmosphere reflectances to be corrected.", schema={"type": "object", "subtype": "raster-cube"})
        .param('collection', description='Region',schema={"type": "string"})
        .returns(description="the corrected data as a data cube", schema={"type": "object", "subtype": "raster-cube"}),
    registries=[bdc_registry,]
)
def bdc_publish_stac(args: Dict, env: EvalEnv) -> object:
    pass


@bdc_non_standard_process(
    ProcessSpec(
        id='bdc_publish_wms',
        description="Crop scene",
        extra={
            "summary": "Publish WMS",
            "categories": ["cubes", "optical"],
            "experimental": True,
            "links": [
                {
                    "rel": "about",
                    "href": "http://brazildatacube.dpi.inpe.br/openeo/info/bdc_publish_wms",
                    "title": "Publish WMS"
                }
            ],
            "exceptions": {
                "DigitalElevationModelInvalid": {
                    "message": "The digital elevation model specified is either not a DEM or can't be used with the data cube given."
                }
            },
        }
    )
        .param('data', description="Data cube containing multi-spectral optical top of atmosphere reflectances to be corrected.", schema={"type": "object", "subtype": "raster-cube"})
        .param('collection', description='Region',schema={"type": "string"})
        .returns(description="the corrected data as a data cube", schema={"type": "object", "subtype": "raster-cube"}),
    registries=[bdc_registry,]
)
def bdc_publish_wms(args: Dict, env: EvalEnv) -> object:
    pass

class BDCConcreteProcessing(ConcreteProcessing):

    # For lazy loading of (global) process registry
    _registry_cache = {}

    def __init__(self) -> None:
        super().__init__()

    def get_process_registry(self, api_version: Union[str, ComparableVersion]) -> ProcessRegistry:
        return bdc_registry

    def get_basic_env(self, api_version=None) -> EvalEnv:
        return EvalEnv({
            "backend_implementation": BDCBackendImplementation(processing=self),
            "version": api_version or "1.0.0",  # TODO: get better default api version from somewhere?
        })

    def evaluate(self, process_graph: dict, env: EvalEnv = None):
        return evaluate(process_graph=process_graph, env=env or self.get_basic_env(), do_dry_run=False)