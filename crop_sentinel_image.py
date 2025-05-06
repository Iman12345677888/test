import os
import glob
import numpy as np
from osgeo import gdal, osr
import matplotlib.pyplot as plt
from pyproj import Transformer


class CropDatasetProcessor:
    def __init__(self, data_list, out_path, bbox=None):
        self.out_path = out_path
        self.resampled_data = None
        self.data_list = data_list
        self.bbox = self.set_bbox(bbox)
        self.target_res = 10

    def set_bbox(self, bbox):
        if bbox is None:
            for image_path in self.data_list:
                files = os.listdir(os.path.join(image_path, "R10m"))
                file_paths = [f for f in files if os.path.isfile(os.path.join(image_path, "R10m", f))]
                tmp = self.get_bbox(os.path.join(image_path, "R10m", file_paths[0]))
                if bbox is None:
                    bbox = tmp
                else:
                    bbox = (max(bbox[0], tmp[0]), max(bbox[1], tmp[1]), min(bbox[2], tmp[2]), min(bbox[3], tmp[3]))
        else:
            image_path = self.data_list[0]
            files = os.listdir(os.path.join(image_path, "R10m"))
            file_paths = [f for f in files if os.path.isfile(os.path.join(image_path, "R10m", f))]
            dst_crs = self.get_epsg_from_raster(os.path.join(image_path, "R10m", file_paths[0]))
            bbox = self.transform_bbox(bbox, src_crs="EPSG:4326", dst_crs=dst_crs)
            print(bbox)
        return bbox

    def get_epsg_from_raster(self, file_path):
        """Extracts EPSG code from a raster file in the format 'EPSG:XXXX'"""
        try:
            ds = gdal.Open(file_path)
            if ds is None:
                raise ValueError("Could not open raster file")

            srs = osr.SpatialReference()
            srs.ImportFromWkt(ds.GetProjection())

            # Get authority code (EPSG number)
            epsg_code = srs.GetAuthorityCode(None)

            if epsg_code:
                return f"EPSG:{epsg_code}"
            else:
                return None  # No EPSG code found

        except Exception as e:
            print(f"Error reading EPSG: {str(e)}")
            return None

    def transform_bbox(self, bbox, src_crs="EPSG:32632", dst_crs="EPSG:4326"):
        """
        Transform a bounding box from one coordinate system to another.
        """
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        minx, miny = transformer.transform(bbox[0], bbox[1])
        maxx, maxy = transformer.transform(bbox[2], bbox[3])
        return (minx, miny, maxx, maxy)

    def resample_to_30m(self, file_path, out_path="/vsimem/resampled.tif"):
        """Resamples input raster to 30m resolution using GDAL"""
        ds = gdal.Open(file_path)
        warp_options = gdal.WarpOptions(
            xRes=self.target_res,
            yRes=self.target_res,
            resampleAlg='bilinear',  # Can use 'average', 'cubic', etc. depending on needs
            outputBounds=self.bbox  # Expects [minX, minY, maxX, maxY]
        )
        resampled_ds = gdal.Warp(out_path, ds, options=warp_options)
        return resampled_ds

    def resample_batch(self, band_files):
        """Resample a list of band files to 30m and return as numpy array"""
        arrays = []
        for file in band_files:
            print(file)
            resampled_ds = self.resample_to_30m(file)
            band_array = resampled_ds.ReadAsArray()
            arrays.append(band_array)
        return arrays  # np.stack(arrays)

    def get_image_bands(self):
        # image_list = list_tif_files(root_path)
        # bbox = transform_bbox(bbox, src_crs="EPSG:4326", dst_crs="EPSG:32632")
        resampled_data = []
        # for image_path in self.data_list[0:1]:
        image_path = self.data_list
        files = os.listdir(os.path.join(image_path, "R10m"))
        file_paths = [f for f in files if os.path.isfile(os.path.join(image_path, "R10m", f))]
        band_list = ['B04', 'B03', 'B02', 'B08', 'B11']  # ["B04", "B03", "B02", "B8A", "B11", "B12"]
        band_path = [os.path.join(image_path, "R10m", f) for f in file_paths for temp in band_list if temp in f]
        print(band_path)
        files = os.listdir(os.path.join(image_path, "R20m"))
        file_paths = [f for f in files if os.path.isfile(os.path.join(image_path, "R20m", f))]
        band_path = band_path + [os.path.join(image_path, "R20m", f) for f in file_paths for temp in band_list[4:]
                                 if
                                 temp in f]
        band_path = sorted(band_path, key=lambda x: band_list.index([b for b in band_list if b in x][0]))

        resampled_data = resampled_data + self.resample_batch(band_path)

        self.resampled_data = np.stack(resampled_data)

    # def get_file_list(self):
    #     """Get all file paths for data chips"""
    #     return glob.glob(os.path.join(self.dataset_root, "**/*.tif"), recursive=True)

    def get_bbox(self, file_path):
        """Get bounding box from a GeoTIFF"""
        ds = gdal.Open(file_path)
        gt = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize

        minx = gt[0]
        maxx = gt[0] + cols * gt[1]
        miny = gt[3] + rows * gt[5]
        maxy = gt[3]

        return (minx, miny, maxx, maxy)

    def read_and_stack_bands(self, file_path):
        """Read and stack all bands from a GeoTIFF"""
        ds = gdal.Open(file_path)
        bands = []
        for i in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(i).ReadAsArray()
            bands.append(band)
        return np.stack(bands)

    def write_18_band_image_to_vsmem(self, data_array, geotransform=None, projection=None,
                                     out_path="/vsimem/stacked_30m.tif"):
        """
        Write a 224x224x18 array to a /vsimem GeoTIFF with 30m resolution.

        Parameters:
        - data_array: np.ndarray of shape (18, 224, 224)
        - geotransform: tuple or list (optional), GDAL geotransform
        - projection: WKT string or None (optional), GDAL projection
        - out_path: string, path in /vsimem (default: /vsimem/stacked_30m.tif)

        Returns:
        - GDAL dataset written to /vsimem
        """
        if data_array.shape[0] != 18 or data_array.shape[1:] != (224, 224):
            raise ValueError("Input array must have shape (18, 224, 224)")

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(out_path, 224, 224, 18, gdal.GDT_Float32)

        # Default geotransform: top-left at (0, 0), 30m pixels
        if geotransform is None:
            geotransform = (0, 30, 0, 0, 0, -30)  # (originX, pixelWidth, 0, originY, 0, -pixelHeight)

        out_ds.SetGeoTransform(geotransform)

        # Default projection: WGS84 UTM zone 15N (can be replaced)
        if projection is None:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(32615)  # UTM zone 15N
            projection = srs.ExportToWkt()

        out_ds.SetProjection(projection)

        for i in range(18):
            out_ds.GetRasterBand(i + 1).WriteArray(data_array[i])

        out_ds.FlushCache()
        return out_path

    def write_to_vsimem(self, data_array, template_file_path, out_path="/vsimem/output.tif"):
        """Write numpy array to /vsimem using template's geotransform and projection"""
        template_ds = gdal.Open(template_file_path)
        driver = gdal.GetDriverByName('GTiff')
        if len(data_array.shape) > 2:
            out_ds = driver.Create(out_path, data_array.shape[2], data_array.shape[1], data_array.shape[0],
                                   gdal.GDT_Float32)
            out_ds.SetGeoTransform(template_ds.GetGeoTransform())
        else:
            out_ds = driver.Create(out_path, data_array.shape[1], data_array.shape[0], 1,
                                   gdal.GDT_Byte)

            geotransform = template_ds.GetGeoTransform()
            geotransform = (self.bbox[0], 30, geotransform[2], self.bbox[3], geotransform[4],
                            -30)  # template_ds.GetGeoTransform() geotransform[0] geotransform[3]
            out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(template_ds.GetProjection())
        if len(data_array.shape) > 2:
            for i in range(data_array.shape[0]):
                out_ds.GetRasterBand(i + 1).WriteArray(data_array[i])
        else:
            out_ds.GetRasterBand(1).WriteArray(data_array)

        out_ds.FlushCache()
        return out_path

    # Function to process large images


# import argparse
# import ast
# import sys
#
#
# def main(image_list, out_path, bbox):
#     processor = CropDatasetProcessor(image_list, out_path, bbox)
#     processor.get_image_bands()


def crop_sentinel_image(bbox: tuple) -> np.ndarray:
    """
    Crop a Sentinel-2 multi-band image to a specified bounding box.

    Args:
        image_path (str): Path to Sentinel-2 image (.jp2 or .tif).
        bbox (tuple): Bounding box start point 'lon_min', 'lat_max'.

    Returns:
        np.ndarray: Cropped image array (bands, height, width).
    """
    image_list = '/DATA/iman58/project/deep_clip/dataset/adana/S2B_MSIL2A_20230525T081609_N0509_R121_T36SYF_20230525T101400.SAFE/GRANULE/L2A_T36SYF_A032465_20230525T082252/IMG_DATA/'
    out_path = ''
    start_lon, start_lat = bbox
    width_px, height_px = 512, 512
    pixel_size = 10  # meters

    # Calculate total width/height in meters
    width_m = width_px * pixel_size  # 5120m
    height_m = height_px * pixel_size  # 5120m

    # Define transformers (WGS84 <-> Web Mercator)
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    to_4326 = Transformer.from_crs("EPSG:32636", "EPSG:4326", always_xy=True)

    # Convert start point to Web Mercator (meters)
    start_x, start_y = to_3857.transform(start_lon, start_lat)

    # Calculate end point (meters)
    end_x = start_x + width_m  # Easting increases
    end_y = start_y - height_m  # Northing decreases (southward)

    # Convert back to WGS84
    end_lon, end_lat = to_4326.transform(end_x, end_y)

    bbox = (bbox[0], bbox[1], end_lon, end_lat)

    processor = CropDatasetProcessor(image_list, out_path, bbox)
    processor.get_image_bands()
    return processor.resampled_data

#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Process crop dataset with image list, output path, and bounding box.')
#     parser.add_argument('--image_list', type=str, required=True,
#                         help='List of image paths as a string representation of a Python list')
#     parser.add_argument('--out_path', type=str, required=True, help='Output path for the processed image')
#     parser.add_argument('--bbox', type=str, required=True,
#                         help='Bounding box as a string representation of a Python tuple (minx, miny, maxx, maxy)')
#
#     args = parser.parse_args()
#
#     try:
#         # Safely evaluate string representations of lists and tuples
#         image_list = ast.literal_eval(args.image_list)
#         out_path = args.out_path
#         bbox = ast.literal_eval(args.bbox)
#
#         start_lon, start_lat = bbox
#         width_px, height_px = 512, 512
#         pixel_size = 10  # meters
#
#         # Calculate total width/height in meters
#         width_m = width_px * pixel_size  # 5120m
#         height_m = height_px * pixel_size  # 5120m
#
#         # Define transformers (WGS84 <-> Web Mercator)
#         to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
#         to_4326 = Transformer.from_crs("EPSG:32636", "EPSG:4326", always_xy=True)
#
#         # Convert start point to Web Mercator (meters)
#         start_x, start_y = to_3857.transform(start_lon, start_lat)
#
#         # Calculate end point (meters)
#         end_x = start_x + width_m  # Easting increases
#         end_y = start_y - height_m  # Northing decreases (southward)
#
#         # Convert back to WGS84
#         end_lon, end_lat = to_4326.transform(end_x, end_y)
#
#         bbox = (bbox[0], bbox[1], end_lon, end_lat)
#
#         # Verify types
#         if not isinstance(image_list, str):
#             raise ValueError("image_list must be a list")
#         if not isinstance(out_path, str):
#             raise ValueError("out_path must be a string")
#         if not isinstance(bbox, tuple) or len(bbox) != 4:
#             raise ValueError("bbox must be a tuple of 4 numbers")
#
#         main(image_list, out_path, bbox)
#
#     except (ValueError, SyntaxError) as e:
#         print(f"Error parsing arguments: {e}")
#         sys.exit(1)
