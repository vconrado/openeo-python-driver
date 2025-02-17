import glob
import os
import pathlib
import tempfile
import warnings
import logging
from abc import ABC
from pathlib import Path
from shutil import copy
from tempfile import mkstemp
from typing import Union, Dict
from zipfile import ZipFile

import numpy as np
import pandas as pd
from flask import send_from_directory, jsonify
from shapely.geometry import GeometryCollection, mapping

from openeo.metadata import CollectionMetadata
from openeo_driver.datacube import DriverDataCube
from openeo_driver.errors import OpenEOApiException
from openeo_driver.utils import replace_nan_values

_log = logging.getLogger(__name__)

class SaveResult(ABC):
    """
    A class that generates a Flask response.
    """

    def __init__(self, format: str = None, options: dict = None):
        self.format = format and format.lower()
        self.options = options or {}

    def is_format(self, *args):
        return self.format.lower() in {f.lower() for f in args}

    def set_format(self, format: str, options: dict = None):
        self.format = format.lower()
        self.options = options or {}

    def create_flask_response(self):
        """
        Returns a Flask compatible response.

        :return: A response that can be handled by Flask
        """
        pass

    def get_mimetype(self, default="application/octet-stream"):
        return {
            "gtiff": "image/tiff; application=geotiff",
            "cog": "image/tiff; application=geotiff; profile=cloud-optimized",
            "netcdf": "application/x-netcdf",
            "png": "image/png",
            "json": "application/json",
            "geojson": "application/geo+json",
            "covjson": "application/json",
            "csv": "text/csv",
            # TODO: support more formats
        }.get(self.format.lower(), default)


def get_temp_file(suffix="", prefix="openeo-pydrvr-"):
    # TODO: make sure temp files are cleaned up when read
    _, filename = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    return filename


class ImageCollectionResult(SaveResult):

    def __init__(self, cube: DriverDataCube, format: str, options: dict):
        super().__init__(format=format, options=options)
        self.cube = cube

    def save_result(self, filename: str) -> str:
        return self.cube.save_result(filename=filename, format=self.format, format_options=self.options)

    def write_assets(self, directory:str) -> Dict:
        """
        Save generated assets into a directory, return asset metadata.
        TODO: can an asset also be a full STAC item? In principle, one openEO job can either generate a full STAC collection, or one STAC item with multiple assets...

        :return: STAC assets dictionary: https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md#assets
        """
        if "write_assets" in dir(self.cube):
            return self.cube.write_assets(filename=directory, format=self.format, format_options=self.options)
        else:
            filename = self.cube.save_result(filename=directory, format=self.format, format_options=self.options)
            return {filename:{"href":filename}}

    def create_flask_response(self):
        filename = get_temp_file(suffix=".save_result.{e}".format(e=self.format.lower()))
        filename = self.save_result(filename)
        mimetype = self.get_mimetype()
        return send_from_directory(os.path.dirname(filename), os.path.basename(filename), mimetype=mimetype)


class JSONResult(SaveResult):

    def __init__(self, data, format: str = "json", options: dict = None):
        super().__init__(format=format, options=options)
        self.data = data

    def write_assets(self, path:str) -> Dict:
        """
        Save generated assets into a directory, return asset metadata.
        TODO: can an asset also be a full STAC item? In principle, one openEO job can either generate a full STAC collection, or one STAC item with multiple assets...

        :return: STAC assets dictionary: https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md#assets
        """
        output_dir = Path(path).parent
        output_file = output_dir / "result.json"
        with open(output_file, 'w') as f:
            import json
            json.dump(self.prepare_for_json(), f)
        return {"result.json":{
            "href":str(output_file),
            "roles": ["data"],
            "type": "application/json",
            "description": "json result generated by openEO"
            }}

    def get_data(self):
        return self.data

    def prepare_for_json(self):
        return replace_nan_values(self.get_data())

    def create_flask_response(self):
        return jsonify(self.prepare_for_json())


class AggregatePolygonResult(JSONResult):
    """
    Container for timeseries result of `aggregate_polygon` process (aka "zonal stats")

    Expects internal representation of timeseries as nested structure:

        dict mapping timestamp (str) to:
            a list, one item per polygon:
                a list, one float per band

    """

    def __init__(self, timeseries: dict, regions: GeometryCollection, metadata:CollectionMetadata=None):
        super().__init__(data=timeseries)
        if not isinstance(regions, GeometryCollection):
            # TODO: raise exception instead of warning?
            warnings.warn("AggregatePolygonResult: GeometryCollection expected but got {t}".format(t=type(regions)))
        self._regions = regions
        self._metadata = metadata

    def get_data(self):
        if self.is_format('covjson', 'coveragejson'):
            return self.to_covjson()
        # By default, keep original (proprietary) result format
        return self.data

    def write_assets(self, directory:str) -> Dict:
        """
        Save generated assets into a directory, return asset metadata.
        TODO: can an asset also be a full STAC item? In principle, one openEO job can either generate a full STAC collection, or one STAC item with multiple assets...

        :return: STAC assets dictionary: https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md#assets
        """
        directory = pathlib.Path(directory).parent
        filename = str(Path(directory)/"timeseries.json")
        asset = {
            "roles": ["data"],
            "type": "application/json"
        }
        if self.is_format('netcdf', 'ncdf'):
            filename = str(Path(directory) / "timeseries.nc")
            self.to_netcdf(filename)
            asset["type"] = "application/x-netcdf"
        elif self.is_format('csv'):
            filename = str(Path(directory) / "timeseries.csv")
            self.to_csv(filename)
            asset["type"] = "text/csv"
        else:
            import json
            with open(filename, 'w') as f:
                json.dump(self.prepare_for_json(), f)
        asset["href"] = filename
        if self._metadata is not None and self._metadata.has_band_dimension():
            bands = [b._asdict() for b in self._metadata.bands]
            asset["bands"] = bands

        return {str(Path(filename).name): asset}

    def create_flask_response(self):
        if self.is_format('netcdf', 'ncdf'):
            filename = self.to_netcdf()
            return send_from_directory(os.path.dirname(filename), os.path.basename(filename))

        if self.is_format('csv'):
            filename = self.to_csv()
            return send_from_directory(os.path.dirname(filename), os.path.basename(filename))

        return super().create_flask_response()

    def create_point_timeseries_xarray(self, feature_ids, timestamps,lats,lons,averages_by_feature):
        import xarray as xr
        import pandas as pd

        #xarray breaks with timezone aware dates: https://github.com/pydata/xarray/issues/1490
        time_index = pd.to_datetime(timestamps,utc=False).tz_convert(None)
        if len(averages_by_feature.shape) == 3:
            array_definition = {'t': time_index}
            for band in range(averages_by_feature.shape[2]):
                array_definition['band_%s'%str(band)] =  (('feature', 't'), averages_by_feature[:, :, band])
            the_array = xr.Dataset(array_definition)
        else:
            the_array = xr.Dataset({
                'avg': (('feature', 't'), averages_by_feature),
                't': time_index})

        the_array = the_array.dropna(dim='t',how='all')
        the_array = the_array.sortby('t')

        the_array.coords['lat'] = (('feature'),lats)
        the_array.coords['lon'] = (('feature'), lons)
        the_array.coords['feature_names'] = (('feature'), feature_ids)

        the_array.variables['lat'].attrs['units'] = 'degrees_north'
        the_array.variables['lat'].attrs['standard_name'] = 'latitude'
        the_array.variables['lon'].attrs['units'] = 'degrees_east'
        the_array.variables['lon'].attrs['standard_name'] = 'longitude'
        the_array.variables['t'].attrs['standard_name'] = 'time'
        the_array.attrs['Conventions'] = 'CF-1.8'
        the_array.attrs['source'] = 'Aggregated timeseries generated by openEO GeoPySpark backend.'
        return the_array

    def to_netcdf(self,destination = None):
        points = [r.representative_point() for r in self._regions]
        lats = [p.y for p in points]
        lons = [p.x for p in points]
        feature_ids = ['feature_%s'% str(i) for i in range(len(self._regions))]

        values = self.data.values()
        cleaned_values = [[bands if len(bands)>0 else [np.nan,np.nan] for bands in feature_bands ] for feature_bands in values]
        time_feature_bands_array = np.array(list(cleaned_values))
        #time_feature_bands_array[time_feature_bands_array == []] = [np.nan, np.nan]
        assert len(feature_ids) == time_feature_bands_array.shape[1]
        #if len(time_feature_bands_array.shape) == 3:
        #    nb_bands = time_feature_bands_array.shape[2]
        feature_time_bands_array = np.swapaxes(time_feature_bands_array,0,1)
        array = self.create_point_timeseries_xarray(feature_ids, list(self.data.keys()), lats, lons,feature_time_bands_array)
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in array.data_vars}
        if destination is None:
            filename = get_temp_file(suffix=".netcdf")
        else:
            filename = destination
        array.to_netcdf(filename,encoding=encoding)
        return filename

    def to_csv(self, destination=None):
        nb_bands = max([len(item) for sublist in self.data.values() for item in sublist])

        date_band_dict = {}

        for date, polygon_results in self.data.items():
            if nb_bands > 1:
                for i in range(0, nb_bands):
                    date_band = date + '__' + str(i + 1).zfill(2)
                    date_band_dict[date_band] = [(p[i] if p else None) for p in polygon_results]
            else:
                date_band_dict[date] = [(p[0] if p else None) for p in polygon_results]

        if destination is None:
            filename = get_temp_file(suffix=".csv")
        else:
            filename = destination

        pd.DataFrame(date_band_dict).to_csv(filename, index=False)

        return filename

    def to_covjson(self) -> dict:
        """
        Convert internal timeseries structure to Coverage JSON structured dict
        """

        # Convert GeometryCollection to list of GeoJSON Polygon coordinate arrays
        polygons = [p["coordinates"] for p in mapping(self._regions)["geometries"]]

        # TODO make sure timestamps are ISO8601 (https://covjson.org/spec/#temporal-reference-systems)
        timestamps = sorted(self.data.keys())

        # Count bands in timestamp data
        # TODO get band count and names from metadata
        band_counts = set(len(polygon_data) for ts_data in self.data.values() for polygon_data in ts_data)
        band_counts.discard(0)
        if len(band_counts) != 1:
            raise ValueError("Multiple band counts in data: {c}".format(c=band_counts))
        band_count = band_counts.pop()

        # build parameter value array (one for each band)
        actual_timestamps = []
        param_values = [[] for _ in range(band_count)]
        for ts in timestamps:
            ts_data = self.data[ts]
            if len(ts_data) != len(polygons):
                warnings.warn("Expected {e} polygon results, but got {g}".format(e=len(polygons), g=len(ts_data)))
                continue
            if all(len(polygon_data) != band_count for polygon_data in ts_data):
                # Skip timestamps without any complete data
                # TODO: also skip timestamps with only NaNs?
                continue
            actual_timestamps.append(ts)
            for polygon_data in ts_data:
                if len(polygon_data) == band_count:
                    for b, v in enumerate(polygon_data):
                        param_values[b].append(v)
                else:
                    for b in range(band_count):
                        param_values[b].append(np.nan)

        domain = {
            "type": "Domain",
            "domainType": "MultiPolygonSeries",
            "axes": {
                "t": {"values": actual_timestamps},
                "composite": {
                    "dataType": "polygon",
                    "coordinates": ["x", "y"],
                    "values": polygons,
                },
            },
            "referencing": [
                {
                    "coordinates": ["x", "y"],
                    "system": {
                        "type": "GeographicCRS",
                        "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",  # TODO check CRS
                    }
                },
                {
                    "coordinates": ["t"],
                    "system": {"type": "TemporalRS", "calendar": "Gregorian"}
                }
            ]
        }
        parameters = {
            "band{b}".format(b=band): {
                "type": "Parameter",
                "observedProperty": {"label": {"en": "Band {b}".format(b=band)}},
                # TODO: also add unit properties?
            }
            for band in range(band_count)
        }
        shape = (len(actual_timestamps), len(polygons))
        ranges = {
            "band{b}".format(b=band): {
                "type": "NdArray",
                "dataType": "float",
                "axisNames": ["t", "composite"],
                "shape": shape,
                "values": param_values[band],
            }
            for band in range(band_count)
        }
        return {
            "type": "Coverage",
            "domain": domain,
            "parameters": parameters,
            "ranges": ranges
        }



class AggregatePolygonResultCSV(AggregatePolygonResult):


    def __init__(self, csv_dir, regions: GeometryCollection, metadata:CollectionMetadata=None):

        def _flatten_df(df):
            df.index = df.feature_index
            df.sort_index(inplace=True)
            return df.drop(columns="feature_index").values.tolist()

        paths = list(glob.glob(os.path.join(csv_dir, "*.csv")))
        _log.info(f"Parsing intermediate timeseries results: {paths}")
        if(len(paths)==0):
            raise OpenEOApiException(status_code=500, code="EmptyResult", message=f"aggregate_spatial did not generate any output, intermediate output path on the server: {csv_dir}")
        df = pd.concat(map(pd.read_csv, paths))

        super().__init__(timeseries={pd.to_datetime(date).tz_convert('UTC').strftime('%Y-%m-%dT%XZ'): _flatten_df(df[df.date == date].drop(columns="date")) for date in df.date.unique()},regions=regions,metadata=metadata)
        self._csv_dir = csv_dir

    def to_csv(self, destination=None):
        csv_paths = glob.glob(self._csv_dir + "/*.csv")
        print(csv_paths)
        if(destination == None):
            return csv_paths[0]
        else:
            copy(csv_paths[0],destination)
            return destination



class MultipleFilesResult(SaveResult):
    def __init__(self, format: str, *files: Path):
        super().__init__(format=format)
        self.files = list(files)

    def reduce(self, output_file: Union[str, Path], delete_originals: bool):
        with ZipFile(output_file, "w") as zip_file:
            for file in self.files:
                zip_file.write(filename=file, arcname=file.name)

        if delete_originals:
            for file in self.files:
                file.unlink()

    def create_flask_response(self):
        from flask import current_app

        _, temp_file_name = mkstemp(suffix=".zip", dir=Path.cwd())
        temp_file = Path(temp_file_name)

        def stream_and_remove():
            self.reduce(temp_file, delete_originals=True)

            with open(temp_file, "rb") as f:
                yield from f

            temp_file.unlink()

        resp = current_app.response_class(stream_and_remove(), mimetype="application/zip")
        resp.headers.set('Content-Disposition', 'attachment', filename=temp_file.name)

        return resp


class NullResult(SaveResult):
    def __init__(self):
        super().__init__()

    def create_flask_response(self):
        return jsonify(None)

