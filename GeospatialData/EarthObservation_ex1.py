import numpy as np
from osgeo import gdal, osr
from pyproj import Proj, Transformer
from PIL import Image


def convert_coords_to_pixel(dataset, x, y):
    """
    Convert geographic coordinates to pixel coordinates.
    """
    geo_transform = dataset.GetGeoTransform()
    proj = dataset.GetProjection()

    # Create a spatial reference object from the dataset projection
    dataset_srs = osr.SpatialReference()
    dataset_srs.ImportFromWkt(proj)

    # Initialize the dataset CRS and WGS84 CRS
    dataset_crs = Proj(proj)
    wgs84 = Proj("EPSG:4326")

    # Create a transformer object
    transformer = Transformer.from_proj(wgs84, dataset_crs, always_xy=True)

    # Transform the coordinates
    x_proj, y_proj = transformer.transform(x, y)

    # Calculate pixel and line
    pixel = int((x_proj - geo_transform[0]) / geo_transform[1])
    line = int((y_proj - geo_transform[3]) / geo_transform[5])

    return pixel, line

def crop_image(dataset, center_lon, center_lat, width, height):
    """
    Crop the image around the specified center coordinates.
    """
    pixel, line = convert_coords_to_pixel(dataset, center_lon, center_lat)
    
    half_width = width // 2
    half_height = height // 2

    x_offset = max(0, pixel - half_width)
    y_offset = max(0, line - half_height)
    x_end = min(dataset.RasterXSize, pixel + half_width)
    y_end = min(dataset.RasterYSize, line + half_height)

    x_size = max(0, x_end - x_offset)
    y_size = max(0, y_end - y_offset)

    if x_size == 0 or y_size == 0:
        raise ValueError("Cropped area dimensions are invalid (zero size)")

    cropped_data = dataset.ReadAsArray(x_offset, y_offset, x_size, y_size)

    return cropped_data, x_offset, y_offset, dataset.GetGeoTransform()

def save_cropped_image(cropped_data, x_offset, y_offset, geo_transform, projection, output_path):
    """
    Save the cropped image to a file.
    """
    driver = gdal.GetDriverByName('GTiff')

    _, height, width, num_bands = cropped_data.shape

    out_dataset = driver.Create(output_path, width, height, num_bands, gdal.GDT_Byte)

    new_geo_transform = list(geo_transform)
    new_geo_transform[0] += x_offset * geo_transform[1]
    new_geo_transform[3] += y_offset * geo_transform[5]

    out_dataset.SetGeoTransform(new_geo_transform)
    out_dataset.SetProjection(projection)

    out_dataset.GetRasterBand(1).WriteArray(cropped_data[0, :, :, 0])      #red
    out_dataset.GetRasterBand(2).WriteArray(cropped_data[0, :, :, 1])      #green
    out_dataset.GetRasterBand(3).WriteArray(cropped_data[0, :, :, 2])      #blue

    # close data set
    out_dataset.FlushCache()
    out_dataset = None

def brigthness(output_path, factor):
    """
    Increase the brightness by a factor (%) of a given .tiff file
    """
    
    # Open the TIFF image
    rgb_tiff = Image.open(output_path)

    # Facteur d'augmentation de luminosité
    brightness_factor = 1 + factor/100  # Augmentation de 110%

    # Charger l'image RGB créée précédemment
    rgb_image = np.array(rgb_tiff)

    # Convertir l'image en tableau NumPy
    rgb_array = np.array(rgb_image)

    # Augmentation de la luminosité en multipliant toutes les valeurs des pixels par un facteur
    brightened_rgb_array = np.clip(rgb_array * brightness_factor, 0, 255).astype(np.uint8)

    # Enregistrer l'image augmentée de luminosité
    brightened_image = Image.fromarray(brightened_rgb_array)
    brightened_image.save('Bright_'+output_path)

    print("L'image avec une luminosité augmentée a été enregistrée.")

def CREATE_CROPPED_IMAGE(band_1, band_2, band_3, center_lon, center_lat, pixel_size, crop_width, crop_height, path ):
    # File paths
    band_1 = 'EO_Browser_images_uint16/2022-04-16-00[]00_2022-04-16-23[]59_Sentinel-2_L2A_B02_(Raw).tiff'
    band_2 = 'EO_Browser_images_uint16/2022-04-16-00[]00_2022-04-16-23[]59_Sentinel-2_L2A_B03_(Raw).tiff'
    band_3 = 'EO_Browser_images_uint16/2022-04-16-00[]00_2022-04-16-23[]59_Sentinel-2_L2A_B04_(Raw).tiff'

    # Read spectral bands
    band1_dataset = gdal.Open(band_1)
    band2_dataset = gdal.Open(band_2)
    band3_dataset = gdal.Open(band_3)

    # Crop the images
    band1_cropped, x_offset, y_offset, geo_transform = crop_image(band1_dataset, center_lon, center_lat, crop_width, crop_height)
    band2_cropped, _, _, _                           = crop_image(band2_dataset, center_lon, center_lat, crop_width, crop_height)
    band3_cropped, _, _, _                           = crop_image(band3_dataset, center_lon, center_lat, crop_width, crop_height)

    # Stack the bands to create an RGB image
    rgb_cropped = np.stack((band3_cropped, band2_cropped, band1_cropped), axis=-1)

    # Normalisation des valeurs de l'image pour être entre 0 et 255
    rgb_cropped = ((rgb_cropped - np.min(rgb_cropped)) / (np.max(rgb_cropped) - np.min(rgb_cropped))) * 255
    rgb_cropped = rgb_cropped.astype(np.uint8)

    # Save the cropped RGB image
    output_path = 'Crop_'+path
    projection = band1_dataset.GetProjection()
    save_cropped_image(rgb_cropped, x_offset, y_offset, geo_transform, projection, output_path)

    print(f"The cropped RGB image has been created and saved as {output_path}")

    # Close the datasets
    band1_dataset = None
    band2_dataset = None
    band3_dataset = None

    return output_path

def RGB_3bands(band_1, band_2, band_3):
    """
    • Takes 3 spectral bands from the 12 bands of Sentinel-2
    • Open, Stack and Creates an RGB image. 
    • Band_k is a string of the form "B01","B02",...,"B11"
    • Returns an output path in order to be enhanced later by other methods
    """

    # paths the files with given spectral band
    band1_path = 'EO_Browser_images_uint16/2022-04-16-00[]00_2022-04-16-23[]59_Sentinel-2_L2A_'+str(band_1)+'_(Raw).tiff'
    band2_path = 'EO_Browser_images_uint16/2022-04-16-00[]00_2022-04-16-23[]59_Sentinel-2_L2A_'+str(band_2)+'_(Raw).tiff'
    band3_path = 'EO_Browser_images_uint16/2022-04-16-00[]00_2022-04-16-23[]59_Sentinel-2_L2A_'+str(band_3)+'_(Raw).tiff'
    
    # Open with GDAL
    band1_dataset = gdal.Open(band1_path)
    band2_dataset = gdal.Open(band2_path)
    band3_dataset = gdal.Open(band3_path)

    band1 = band1_dataset.GetRasterBand(1).ReadAsArray()
    band2 = band2_dataset.GetRasterBand(1).ReadAsArray()
    band3 = band3_dataset.GetRasterBand(1).ReadAsArray()

    # check band dimensions of the 3 files are the same
    assert band1.shape == band2.shape == band3.shape, "Les dimensions des bandes ne correspondent pas."

    # Stack to form RGB image
    rgb_image = np.dstack((band3, band2, band1))

    # Normalize to get values between 0 et 255
    rgb_image = ((rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))) * 255
    rgb_image = rgb_image.astype(np.uint8)

    # Define the output file
    driver = gdal.GetDriverByName('GTiff')

    # Create it
    out_path = 'RGB.tiff'
    out_dataset = driver.Create(out_path, band1_dataset.RasterXSize, band1_dataset.RasterYSize, 3, gdal.GDT_Byte)

    # Define geogrphic transform & projection 
    out_dataset.SetGeoTransform(band1_dataset.GetGeoTransform())
    out_dataset.SetProjection(band1_dataset.GetProjection())

    # write the band in the output dataset
    out_dataset.GetRasterBand(1).WriteArray(rgb_image[:, :, 0])  # Red
    out_dataset.GetRasterBand(2).WriteArray(rgb_image[:, :, 1])  # Green
    out_dataset.GetRasterBand(3).WriteArray(rgb_image[:, :, 2])  # Blue

    # Close datasets
    out_dataset.FlushCache()
    band1_dataset = None
    band2_dataset = None
    band3_dataset = None
    out_dataset = None

    print(f"L'image RGB a été créée et enregistrée sous {out_path}")

    return out_path

def main():

    B02 = 'B02' 
    B03 = 'B03'
    B04 = 'B04'

    # Coordinates of the center point
    center_lon = 8.888669
    center_lat = 44.426111

    # Dimensions of the cropped area in pixels (2 km x 2 km)
    pixel_size = 10  # Resolution of the bands (10 m per pixel for Sentinel-2 for B02,B03 & B04)
    crop_width = 2000 // pixel_size  # 2 km in pixels
    crop_height = 2000 // pixel_size  # 2 km in pixels

    path = RGB_3bands(B02, B03, B04)
    brigthness(path, 200)
    output_path = CREATE_CROPPED_IMAGE(B02, B03, B04, center_lon, center_lat, pixel_size, crop_width, crop_height, path)
    brigthness(output_path, 200)

if __name__ == "__main__":
    main()

