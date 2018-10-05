from unittest.mock import Mock
from openeo import ImageCollection
import os
from shapely.geometry import Polygon, MultiPolygon


def getImageCollection(product_id, viewingParameters):
    image_collection = ImageCollection()
    image_collection.viewingParameters = viewingParameters

    if product_id == 'S2_FAPAR_CLOUDCOVER':
        download = Mock(name='download')
        download.return_value = os.path.realpath(__file__)

        image_collection.download = download
    else:
        timeseries = Mock(name='timeseries')
        timeseries.return_value = {
            "viewingParameters" : image_collection.viewingParameters
        }

        image_collection.timeseries = timeseries

        def is_polygon_or_multipolygon(return_value, regions, func):
            assert func == 'mean' or func == 'avg'
            assert isinstance(regions, Polygon) or isinstance(regions, MultiPolygon)
            return return_value

        zonal_statistics = Mock(name='zonal_statistics')
        zonal_statistics.side_effect = lambda regions, func: is_polygon_or_multipolygon({'hello': 'world'}, regions, func)

        image_collection.zonal_statistics = zonal_statistics

        image_collection.apply_pixel = Mock(name = "apply_pixel")
        image_collection.apply_pixel.return_value = image_collection

    return image_collection


fapar_layer = {'product_id': 'S2_FAPAR_CLOUDCOVER'}
def get_layers():
    return [fapar_layer]

def get_layer(product_id):
    if product_id == 'S2_FAPAR_CLOUDCOVER':
        return fapar_layer
    else:
        raise ValueError("Unknown collection: " + product_id)


def health_check():
    return "OK"


def create_batch_job(*_):
    return '07024ee9-7847-4b8a-b260-6c879a2b3cdc'


def run_batch_job(*_):
    return


def get_batch_job_info(job_id):
    return {
        'job_id': job_id,
        'status': 'running'
    }


def get_batch_job_result_filenames(job_id):
    pass


def get_batch_job_result_output_dir(job_id):
    return "/path/to/%s" % job_id
