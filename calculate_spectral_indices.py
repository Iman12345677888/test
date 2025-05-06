import numpy as np

import spyndex
import xarray as xr
import rasterio
# from rasterio.plot import show
import matplotlib.pyplot as plt
import json

class IndexCalculate():
    def __init__(self):
        self.interpretations = None
        self.image_level_indexes = None
        self.indices = None


    # Function to load specific bands from a multi-band TIFF file
    def load_bands_from_tiff(self,tiff_path, band_indices, band_names):
        with rasterio.open(tiff_path) as src:
            # Read the specified bands (band indices start at 1 in rasterio)
            bands = [src.read(band) for band in band_indices]
            # Convert to xarray.DataArray
        return xr.DataArray(
            bands,
            dims=("band", "y", "x"),
            coords={"band": [band_names[idx - 1] for idx in band_indices]},
        )

    def load_xr_from_np(self,bands_np, band_indices, band_names):
        bands = [bands_np[i] for i in range(len(band_indices))]
        # Convert to xarray.DataArray
        return xr.DataArray(
            bands,
            dims=("band", "y", "x"),
            coords={"band": [band_names[idx - 1] for idx in band_indices]},
        )

    def interpret_ndvi(self,ndvi):
        if ndvi is None:
            return "No data available"
        if ndvi < 0:
            return "Non-vegetated surface (water, barren land)"
        elif 0 <= ndvi < 0.2:
            return "Sparse vegetation or stressed plants"
        elif 0.2 <= ndvi < 0.5:
            return "Moderate vegetation"
        else:
            return "Dense, healthy vegetation"

    def interpret_ndwi(self,ndwi):
        if ndwi is None:
            return "No data available"
        return "Water body" if ndwi > 0 else "Non-water surface"

    def interpret_evi(self,evi):
        if evi is None:
            return "No data available"
        return "Healthier vegetation" if evi > 0.2 else "Sparse or stressed vegetation"

    def interpret_savi(self,savi):
        if savi is None:
            return "No data available"
        return "Healthy vegetation" if savi > 0.3 else "Sparse vegetation"

    def interpret_ndbi(self,ndbi):
        if ndbi is None:
            return "No data available"
        return "Built-up area" if ndbi > 0 else "Non-built-up area"

    def interpret_ndmi(self,ndmi):
        if ndmi is None:
            return "No data available"
        return "Higher vegetation moisture content" if ndmi > 0 else "Lower moisture content"

    def interpret_arvi(self,arvi):
        if arvi is None:
            return "No data available"
        return "Vegetation presence with atmospheric correction" if arvi > 0.2 else "Low vegetation cover"

    def interpret_gndvi(self,gndvi):
        if gndvi is None:
            return "No data available"
        return "Higher chlorophyll content" if gndvi > 0.3 else "Lower chlorophyll content"

    def interpret_sipi(self,sipi):
        if sipi is None:
            return "No data available"
        return "Higher carotenoids (plant stress or senescence)" if sipi > 1 else "Healthy plant pigments"

    def interpret_exg(self,exg):
        if exg is None:
            return "No data available"
        return "Green vegetation presence" if exg > 0 else "Non-green surface"

    def interpret_exr(self,exr):
        if exr is None:
            return "No data available"
        return "Red-reflecting surfaces detected" if exr > 0 else "Non-red dominant area"

    def interpret_exgr(self,exgr):
        if exgr is None:
            return "No data available"
        return "Green vegetation detected" if exgr > 0 else "Low green vegetation"

    def interpret_vari(self,vari):
        if vari is None:
            return "No data available"
        return "Vegetation presence in visible spectrum" if vari > 0 else "Low vegetation visibility"

    def compute_index(self,bands):
        # Extract individual bands
        red = bands.sel(band="B04")  # B04 is the Red band
        green = bands.sel(band="B03")  # B03 is the Green band
        blue = bands.sel(band="B02")  # B02 is the Blue band
        nir = bands.sel(band="B08")  # B08 is the NIR band
        swir = bands.sel(band="B11")  # B11 is the SWIR band

        # Scale the data (if necessary, e.g., for Sentinel-2 data)
        red = red / 10000
        green = green / 10000
        blue = blue / 10000
        nir = nir / 10000
        swir = swir / 10000

        # Compute spectral indices using spyndex  "NDRE",
        self.indices = spyndex.computeIndex(
            index=["NDVI", "NDWI", "EVI", "SAVI", "NDBI", "NDMI", "ARVI", "GNDVI", "SIPI", "ExG", "ExR", "ExGR",
                   "VARI"],
            params={
                "N": nir,  # NIR band (B08)
                "R": red,  # Red band (B04)
                "G": green,  # Green band (B03)
                "B": blue,  # Blue band (B02)
                "A": blue,  # Blue band (B02) for SIPI
                "L": 0.5,  # Soil adjustment factor for SAVI
                "S1": swir,  # SWIR band (B11) for NDBI
                "g": 2.5,  # Gain factor for EVI
                "C1": 6.0,  # Atmospheric resistance coefficient 1 for EVI
                "C2": 7.5,  # Atmospheric resistance coefficient 2 for EVI
                "gamma": 1.0,  # Gamma parameter for ARVI
            }
        )
        return  self.indices

    def pixel_level2image_level(self):
        # Convert pixel-level indexes to image-level indexes (using mean)
        self.image_level_indexes = {}
        for idx_name in self.indices.index.values:
            self.image_level_indexes[idx_name] = float(self.indices.sel(index=idx_name).mean().values)
        return self.image_level_indexes

    def text_interperator(self):
        self.interpretations = {
            "NDVI": self.interpret_ndvi(self.image_level_indexes["NDVI"]),
            "NDWI": self.interpret_ndwi(self.image_level_indexes["NDWI"]),
            "EVI": self.interpret_evi(self.image_level_indexes["EVI"]),
            "SAVI": self.interpret_savi(self.image_level_indexes["SAVI"]),
            "NDBI": self.interpret_ndbi(self.image_level_indexes["NDBI"]),
            "NDMI": self.interpret_ndmi(self.image_level_indexes["NDMI"]),
            "ARVI": self.interpret_arvi(self.image_level_indexes["ARVI"]),
            "GNDVI": self.interpret_gndvi(self.image_level_indexes["GNDVI"]),
            "SIPI": self.interpret_sipi(self.image_level_indexes["SIPI"]),
            "ExG": self.interpret_exg(self.image_level_indexes["ExG"]),
            "ExR": self.interpret_exr(self.image_level_indexes["ExR"]),
            "ExGR": self.interpret_exgr(self.image_level_indexes["ExGR"]),
            "VARI": self.interpret_vari(self.image_level_indexes["VARI"]),
        }
        return self.interpretations

    def json_export(self,file_name):
        # Save interpretations to JSON file
        with open(file_name, "w") as json_file:
            json.dump(self.interpretations, json_file, indent=4)

    def generate_caption(self):
        """
        Generates a caption describing the scene type based on spectral index interpretations.
        """
        if self.interpretations["NDBI"] == "Built-up area":
            return "Urban area"
        elif "Dense, healthy vegetation" in self.interpretations.values():
            return "Dense forest or jungle"
        elif "Moderate vegetation" in self.interpretations.values():
            return "Rural or agricultural land"
        elif self.interpretations["NDWI"] == "Water body":
            return "Water body or wetland"
        elif self.interpretations["NDMI"] == "Lower moisture content" and self.interpretations[
            "NDVI"] == "Sparse vegetation or stressed plants":
            return "Arid or semi-arid land"
        else:
            return "Mixed landscape"








def calculate_spectral_indices(image_array: np.ndarray) -> dict:
    """
    Calculate spectral indices from a Sentinel-2 image array.

    Args:
        image_array (np.ndarray): Multi-band image array (bands, height, width).

    Returns:
        dict: Dictionary of spectral indices (e.g., {'NDVI': array, 'NDWI': array}).
    """


    #######################
    ci = IndexCalculate()
    band_names = ['B04', 'B03', 'B02', 'B08', 'B11']
    band_indices = [1, 2, 3, 4, 5]

    # bands = ci.load_bands_from_tiff(tiff_path, band_indices, band_names)
    bands = ci.load_xr_from_np(image_array, band_indices, band_names)
    ci.compute_index(bands)
    index_values = ci.pixel_level2image_level()


    index_values = replace_non_json_values(index_values)

    return index_values