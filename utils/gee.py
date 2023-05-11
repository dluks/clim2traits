################################
# Earth Engine utils
################################

from pprint import pprint

import ee

ee.Initialize()  # Initialize earth engine


def ee_info(ee_obj):
    pprint(ee_obj.getInfo())


def aggregate_ic(
    ic: ee.ImageCollection, ds: str, de: str, timeframe: str = "month"
) -> ee.ImageCollection:
    """Aggregates an ImageCollection to a designated timeframe (currently only accepts
    "month")

    Args:
        ic (ee.ImageCollection): ImageCollection to be aggregrated
        ds (str): Start date in formate YYYY-mm-dd
        de (str): End date in formate YYYY-mm-dd
        timeframe (str, optional): Time steps with which to aggregate. Defaults to
        "month".

    Returns:
        ee.ImageCollection
    """
    start, end = ee.Date(ds), ee.Date(de)
    ddiff = end.difference(start, timeframe)
    length = ee.List.sequence(0, ddiff.subtract(1))  # => [0, 1, ..., 11]

    def datelist(t):
        return start.advance(t, timeframe)

    dates = length.map(datelist)

    def create_subcollections(t):
        t = ee.Date(t)
        filt_coll = ic.filterDate(t, t.advance(1, timeframe))
        filt_coll = filt_coll.set("bandcount", ee.Number(filt_coll.size())).set(
            "system:time_start", t.millis()
        )
        return filt_coll

    grouped_ic = dates.map(create_subcollections)
    # Filter months with 0 observations
    grouped_ic = grouped_ic.filter(ee.Filter.gt("bandcount", 0))

    # Get bandname
    bn = ee.Image(ic.reduce(ee.Reducer.mean())).bandNames().getInfo()
    try:
        bn = bn[0]
    except IndexError:
        bn = "empty"

    def reduce_mean(grouped_ic: ee.List) -> ee.Image:
        grouped_ic = ee.ImageCollection(grouped_ic)
        t = grouped_ic.get("system:time_start")
        mn = grouped_ic.reduce(ee.Reducer.mean()).set("system:time_start", t).rename(bn)
        # Reset the scale because GEE overwrites the scale of reduced images to 1deg for
        # some reason...
        mn = mn.setDefaultProjection(mn.projection(), scale=1000)
        return mn

    agg_ic = grouped_ic.map(reduce_mean)
    agg_ic = ee.ImageCollection.fromImages(agg_ic)
    agg_ic = agg_ic.filter(ee.Filter.listContains("system:band_names", bn))

    return agg_ic


def bitwise_extract(image: ee.Image, from_bit: int, to_bit: int = None) -> ee.Image:
    """Performs bitwise extraction for masking images from QA bands

    Args:
        image (ee.Image): Single-band image (QA Band) with n-bit values
        from_bit (int): Position of starting bit
        to_bit (int, optional): Position of ending bit (if values span multiple bits).
        Defaults to None.

    Returns:
        ee.Image
    """
    if to_bit is None:
        to_bit = from_bit
    mask_size = ee.Number(1).add(to_bit).subtract(from_bit)
    mask = ee.Number(1).leftShift(mask_size).subtract(1)
    return image.rightShift(from_bit).bitwiseAnd(mask)


def mask_clouds(
    ic: ee.ImageCollection, qa_band: str = "state_1km"
) -> ee.ImageCollection:
    """Masks cloudy (or cloud-shadowed) pixels of MODIS Terra Land Reflectance images
    contained within ImageCollection

    Args:
        ic (ee.ImageCollection): ImageCollection to be masked
        qa_band (str): QA band name

    Returns:
        ee.ImageCollection: Masked image collection
    """

    def mask_clouds_(image: ee.Image) -> ee.Image:
        qa = image.select(qa_band)
        cloud_mask = bitwise_extract(qa, 0, 1).eq(0).Or(bitwise_extract(qa, 0, 1).eq(3))
        # cloud_mask_unset = bitwise_extract(qa, 0, 1).eq(3)
        # cloud_shadow_mask = bitwise_extract(qa, 2).eq(0)
        # cirrus_mask = bitwise_extract(qa, 8, 9).lte(1)

        # mask = (
        #     cloud_shadow_mask.And(cirrus_mask).And(cloud_mask).Or(cloud_mask_unmarked)
        # )
        internal_cloud_mask = bitwise_extract(qa, 10).eq(0)
        mask = internal_cloud_mask.And(cloud_mask)
        image_masked = image.updateMask(mask)

        return image_masked

    return ic.map(lambda image: mask_clouds_(image))


def export_image(
    image: ee.Image, filename: str, folder: str, projection: dict, scale: int
):
    """Export an image to Drive

    Args:
        image (ee.Image): Image to export
        filename (str): Filename to be used as prefix of exported file and name of task
        folder (str): Destination folder in Drive (does not accept nested directories)
        projection (dict): Desired export projection (use getInfo() to get dict)
        scale (int): Scale in meters
    """
    task_config = {
        "fileNamePrefix": filename,
        "folder": folder,
        "crs": projection["crs"],
        "crsTransform": projection["transform"],
        "fileFormat": "GeoTIFF",
        "formatOptions": {"cloudOptimized": True},
        "maxPixels": 1e13,
        "scale": scale,
    }

    task = ee.batch.Export.image.toDrive(image, filename, **task_config)
    task.start()


def export_collection(
    collection: ee.ImageCollection,
    folder: str,
    projection: dict = None,
    scale: int = None,
):
    """Export an ImageCollection to Drive

    Args:
        collection (ee.ImageCollection): ImageCollection to be exported
        folder (str): Destination folder in Drive (does not accept nested directories)
        projection (dict, optional): Desired export projection (use getInfo() to get
        dict). Determined from first image in collection if not provided Defaults to
        None.
        scale (int, optional): Scale in meters. Scale determined from first image in
        collection if not provided. Defaults to None.
    """
    num_images = int(collection.size().getInfo())
    image_list = collection.toList(num_images)
    prefix = collection.first().bandNames().getInfo()[0]

    if not projection:
        projection = collection.first().projection().getInfo()

    if not scale:
        scale = collection.first().projection().nominalScale().int().getInfo()

    for i in range(num_images):
        image = ee.Image(image_list.get(i))
        date = image.get("system:time_start")
        date_name = ee.Date(date).format("YYYY-MM-dd").getInfo()
        out_name = f"{prefix}_{date_name}_{str(scale)}m"
        export_image(image, out_name, folder, projection, scale)
