import geopandas
import pandas as pd
import matplotlib.pyplot as plt
import json
import rasterio
import numpy as np
import cartopy.crs as ccrs

bio_name_map = {
    "BIO1": "Annual Mean Temperature",
    "BIO2": "Mean Diurnal Range",
    "BIO3": "Isothermality",
    "BIO4": "Temperature Seasonality",
    "BIO5": "Max Temperature of Warmest Month",
    "BIO6": "Min Temperature of Coldest Month",
    "BIO7": "Temperature Annual Range",
    "BIO8": "Mean Temperature of Wettest Quarter",
    "BIO9": "Mean Temperature of Driest Quarter",
    "BIO10": "Mean Temperature of Warmest Quarter",
    "BIO11": "Mean Temperature of Coldest Quarter",
    "BIO12": "Annual Precipitation",
    "BIO13": "Precipitation of Wettest Month",
    "BIO14": "Precipitation of Driest Month",
    "BIO15": "Precipitation Seasonality",
    "BIO16": "Precipitation of Wettest Quarter",
    "BIO17": "Precipitation of Driest Quarter",
    "BIO18": "Precipitation of Warmest Quarter",
    "BIO19": "Precipitation of Coldest Quarter"
}

# Set this to True to plot all points as red, or False for split colors
plot_all_red = False  # <--- Change this as needed

long_lat = pd.read_csv("Data/long_lat.csv", sep="\t")
errors = pd.read_csv("Data/prediction_errors_BIO1.csv")

# Drop rows where long or lat is -9
long_lat = long_lat[~((long_lat['LONG'] == -9) | (long_lat['LAT'] == -9))]

with open("Results/splits_test2.json", "r") as f:
        splits = json.load(f)

train_ids = splits.get("train", [])
val_ids = splits.get("val", [])
test_ids = splits.get("test", [])

split_df = pd.DataFrame()

for split, ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
    split_df = pd.concat([split_df, long_lat[long_lat["IID"].isin(ids)].assign(split=split)])


worldmap = geopandas.read_file("D:\\Masters\\Data\\Maps\\ne_10m_land\\ne_10m_land.shp")
counties = geopandas.read_file("D:\\Masters\\Data\\Maps\\ne_10m_admin_0_countries\\ne_10m_admin_0_countries.shp")
# urban = geopandas.read_file("D:\\Masters\\Data\\Maps\\ne_10m_urban_areas\\ne_10m_urban_areas.shp")

# Create figure with North Pole stereographic projection
projection = ccrs.NorthPolarStereo()
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': projection})

# Set extent to focus on data range (add small buffer below min latitude)
min_lat = split_df['LAT'].min()
lat_limit = max(0, min_lat - 5)  # 5 degree buffer, but not below equator
ax.set_extent([-180, 180, lat_limit, 90], ccrs.PlateCarree())

# Plot countries (they are in WGS84/PlateCarree by default)
counties.plot(color="grey", ax=ax, transform=ccrs.PlateCarree())

# --- Add raster overlay ---
# with rasterio.open("Data/WorldClim/wc2.1_2.5m_bio_1.tif") as src:
#     bio3 = src.read(1)
#     bio3 = np.ma.masked_equal(bio3, src.nodata)
#     img = ax.imshow(
#         bio3,
#         extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top),
#         cmap="viridis",
#         alpha=0.5,
#         zorder=1
#     )
#     plt.colorbar(img, ax=ax, label="Isothermality (BIO3)", shrink=0.5, pad=0.02)
# --- End raster overlay ---

if plot_all_red:
    ax.scatter(split_df["LONG"], split_df["LAT"],
               alpha=0.6,
               c="red",
               s=1,
               label="All Points",
               transform=ccrs.PlateCarree())
    # Remove legend if plotting all points
    if ax.get_legend() is not None:
        ax.get_legend().remove()
else:
    for split, color in [("train", "blue"), ("val", "orange"), ("test", "green")]:
        subset = split_df[split_df["split"] == split]
        ax.scatter(subset["LONG"], subset["LAT"],
                   alpha=0.6,
                   c=color,
                   s=2,
                   label=split,
                   transform=ccrs.PlateCarree())
    ax.legend(loc="upper right", fontsize=12)

# Add title
ax.set_title(f"Spatial Distribution of Samples - North Pole View", fontsize=16)

# Save the figure as a high-resolution PNG
plt.savefig('plot_splits.png', dpi=300, bbox_inches='tight')
plt.show()
