import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

"""
Visualization script for 300km buffer coastline + sea ROI filtering results.

Creates multiple visualizations:
1. Global map showing coastline, sea, and land ROIs
2. Regional zoom-ins with different ROI types
3. ROI density heatmap
4. Distribution statistics (pie charts, bar charts)
5. Buffer visualization
6. Excluded ROIs only visualization
"""

# === Paths ===
metadata_file = "/home/wangyu/allclear/metadata/rois/rois_metadata.csv"
coastline_shapefile = "/home/wangyu/CCR/CCR_CoastlineCloudRemoval/coastline_shapefiles/ne_10m_coastline.shp"
land_shapefile = "/home/wangyu/CCR/CCR_CoastlineCloudRemoval/shapefiles/ne_10m_land.shp"
roi_list_file = "/home/wangyu/CCR/CCR_CoastlineCloudRemoval/ROI_filter/300km/coastline_and_sea_rois_300km.txt"
output_dir = "/home/wangyu/CCR/CCR_CoastlineCloudRemoval/ROI_filter/300km/visualizations"

os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("COASTLINE + SEA ROI VISUALIZATION (300km Buffer)")
print("="*80)

# === Load data ===
print("\n1. Loading data...")
df_meta = pd.read_csv(metadata_file)
print(f"   Total ROIs in metadata: {len(df_meta)}")

# Load ROI list
coastline_sea_rois = set()
with open(roi_list_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            coastline_sea_rois.add(line)
print(f"   Coastline + Sea ROIs (300km): {len(coastline_sea_rois)}")

# === Create GeoDataFrame ===
print("\n2. Creating GeoDataFrames...")
gdf_all_rois = gpd.GeoDataFrame(
    df_meta,
    geometry=[Point(lon, lat) for lon, lat in zip(df_meta['longitude'], df_meta['latitude'])],
    crs="EPSG:4326"
)

# Load shapefiles
gdf_coastline = gpd.read_file(coastline_shapefile)
gdf_land = gpd.read_file(land_shapefile)
print(f"   Coastline features: {len(gdf_coastline)}")
print(f"   Land features: {len(gdf_land)}")

# === Identify ROI categories ===
print("\n3. Categorizing ROIs...")

# Buffer coastline
buffer_distance = 2.70  # 300km
gdf_coastline_buffered = gdf_coastline.copy()
gdf_coastline_buffered['geometry'] = gdf_coastline_buffered.geometry.buffer(buffer_distance)

# Spatial joins
gdf_coastline_rois = gpd.sjoin(gdf_all_rois, gdf_coastline_buffered, how="inner", predicate="within")
coastline_roi_ids = set(gdf_coastline_rois['roi_id'].astype(str).tolist())

gdf_land_rois = gpd.sjoin(gdf_all_rois, gdf_land, how="inner", predicate="intersects")
land_roi_ids = set(gdf_land_rois['roi_id'].astype(str).tolist())

all_metadata_rois = set(df_meta['roi_id'].astype(str).tolist())
sea_roi_ids = all_metadata_rois - land_roi_ids
purely_land_rois = land_roi_ids - coastline_roi_ids

# Categorize each ROI
df_meta['roi_id_str'] = df_meta['roi_id'].astype(str)
df_meta['category'] = 'Purely Land (Excluded)'

# Set categories
df_meta.loc[df_meta['roi_id_str'].isin(sea_roi_ids - coastline_roi_ids), 'category'] = 'Sea Only'
df_meta.loc[df_meta['roi_id_str'].isin(coastline_roi_ids - sea_roi_ids), 'category'] = 'Coastline Only'
df_meta.loc[df_meta['roi_id_str'].isin(coastline_roi_ids & sea_roi_ids), 'category'] = 'Coastline + Sea'
df_meta['included'] = df_meta['roi_id_str'].isin(coastline_sea_rois)

print(f"   Coastline ROIs: {len(coastline_roi_ids)}")
print(f"   Sea ROIs: {len(sea_roi_ids)}")
print(f"   Coastline + Sea overlap: {len(coastline_roi_ids & sea_roi_ids)}")
print(f"   Purely Land (excluded): {len(purely_land_rois)}")

# === Visualization 1: Global Overview ===
print("\n4. Creating global overview map...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Left: All ROIs with categories
gdf_land.plot(ax=ax1, color='lightgray', edgecolor='darkgray', linewidth=0.3, alpha=0.5)
gdf_coastline.plot(ax=ax1, color='blue', linewidth=0.5, alpha=0.3)

for category, color, marker, size in [
    ('Purely Land (Excluded)', 'brown', 'x', 20),
    ('Sea Only', 'navy', 'o', 30),
    ('Coastline Only', 'orange', '^', 30),
    ('Coastline + Sea', 'cyan', 's', 35)
]:
    subset = df_meta[df_meta['category'] == category]
    ax1.scatter(subset['longitude'], subset['latitude'], 
                c=color, marker=marker, s=size, alpha=0.6, label=category)

ax1.set_title('All ROIs by Category (300km Buffer)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: Included vs Excluded
gdf_land.plot(ax=ax2, color='lightgray', edgecolor='darkgray', linewidth=0.3, alpha=0.5)
gdf_coastline.plot(ax=ax2, color='blue', linewidth=0.5, alpha=0.3)

included = df_meta[df_meta['included'] == True]
excluded = df_meta[df_meta['included'] == False]

ax2.scatter(excluded['longitude'], excluded['latitude'], 
            c='red', marker='x', s=20, alpha=0.5, label=f'Excluded (Purely Land): {len(excluded)}')
ax2.scatter(included['longitude'], included['latitude'], 
            c='green', marker='o', s=30, alpha=0.6, label=f'Included (Coastline + Sea): {len(included)}')

ax2.set_title('Included vs Excluded ROIs (300km Buffer)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_global_overview.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 01_global_overview.png")
plt.close()

# === Visualization 2: Regional Zoom-ins ===
print("\n5. Creating regional zoom-in maps...")
regions = [
    ('Southeast Asia', (95, 145, -15, 25)),
    ('Europe', (-15, 40, 35, 72)),
    ('East Coast USA', (-85, -65, 25, 45)),
    ('Mediterranean', (-10, 40, 30, 46))
]

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for idx, (region_name, (lon_min, lon_max, lat_min, lat_max)) in enumerate(regions):
    ax = axes[idx]
    
    # Filter data for region
    region_rois = df_meta[
        (df_meta['longitude'] >= lon_min) & (df_meta['longitude'] <= lon_max) &
        (df_meta['latitude'] >= lat_min) & (df_meta['latitude'] <= lat_max)
    ]
    
    # Plot land and coastline
    gdf_land.cx[lon_min:lon_max, lat_min:lat_max].plot(
        ax=ax, color='wheat', edgecolor='darkgray', linewidth=0.5, alpha=0.7
    )
    gdf_coastline.cx[lon_min:lon_max, lat_min:lat_max].plot(
        ax=ax, color='blue', linewidth=1, alpha=0.5
    )
    
    # Plot ROIs by category
    for category, color, marker, size in [
        ('Purely Land (Excluded)', 'brown', 'x', 25),
        ('Sea Only', 'navy', 'o', 35),
        ('Coastline Only', 'orange', '^', 35),
        ('Coastline + Sea', 'cyan', 's', 40)
    ]:
        subset = region_rois[region_rois['category'] == category]
        if len(subset) > 0:
            ax.scatter(subset['longitude'], subset['latitude'], 
                      c=color, marker=marker, s=size, alpha=0.7, label=f'{category} ({len(subset)})')
    
    ax.set_title(f'{region_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_regional_views.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 02_regional_views.png")
plt.close()

# === Visualization 3: Density Heatmap ===
print("\n6. Creating density heatmap...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Included ROIs density
included_rois = df_meta[df_meta['included'] == True]
gdf_land.plot(ax=ax1, color='lightgray', edgecolor='darkgray', linewidth=0.3, alpha=0.3)
gdf_coastline.plot(ax=ax1, color='blue', linewidth=0.5, alpha=0.2)

hexbin1 = ax1.hexbin(included_rois['longitude'], included_rois['latitude'], 
                     gridsize=50, cmap='YlOrRd', alpha=0.7, mincnt=1)
ax1.set_title(f'Included ROIs Density (n={len(included_rois)})', fontsize=14, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
plt.colorbar(hexbin1, ax=ax1, label='ROI Count per Hex')

# Excluded ROIs density
excluded_rois = df_meta[df_meta['included'] == False]
gdf_land.plot(ax=ax2, color='lightgray', edgecolor='darkgray', linewidth=0.3, alpha=0.3)

hexbin2 = ax2.hexbin(excluded_rois['longitude'], excluded_rois['latitude'], 
                     gridsize=50, cmap='Reds', alpha=0.7, mincnt=1)
ax2.set_title(f'Excluded ROIs Density (n={len(excluded_rois)})', fontsize=14, fontweight='bold')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
plt.colorbar(hexbin2, ax=ax2, label='ROI Count per Hex')

plt.tight_layout()
plt.savefig(f'{output_dir}/03_density_heatmap.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 03_density_heatmap.png")
plt.close()

# === Visualization 4: Statistics ===
print("\n7. Creating statistics visualizations...")
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 4a: Category distribution (pie chart)
ax1 = fig.add_subplot(gs[0, 0])
category_counts = df_meta['category'].value_counts()
colors_pie = ['navy', 'orange', 'cyan', 'brown']
ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', 
        colors=colors_pie, startangle=90)
ax1.set_title('ROI Distribution by Category', fontweight='bold')

# 4b: Included vs Excluded (bar chart)
ax2 = fig.add_subplot(gs[0, 1])
inclusion_counts = df_meta['included'].value_counts()
bars = ax2.bar(['Excluded\n(Purely Land)', 'Included\n(Coastline + Sea)'], 
               [inclusion_counts[False], inclusion_counts[True]], 
               color=['red', 'green'], alpha=0.7, edgecolor='black')
ax2.set_ylabel('Number of ROIs')
ax2.set_title('Included vs Excluded ROIs (300km)', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df_meta)*100:.1f}%)',
            ha='center', va='bottom')

# 4c: Venn diagram (text-based)
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
coastline_only = len(coastline_roi_ids - sea_roi_ids)
sea_only = len(sea_roi_ids - coastline_roi_ids)
overlap = len(coastline_roi_ids & sea_roi_ids)
total_included = len(coastline_sea_rois)
total_excluded = len(purely_land_rois)

venn_text = f"""
Set Relationships (300km Buffer):

Coastline Only:  {coastline_only:,}
Sea Only:        {sea_only:,}
Overlap:         {overlap:,}
─────────────────────────
Total Included:  {total_included:,}
Total Excluded:  {total_excluded:,}
─────────────────────────
Grand Total:     {len(df_meta):,}
"""
ax3.text(0.1, 0.5, venn_text, fontsize=12, fontfamily='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax3.set_title('Set Overlap Statistics', fontweight='bold')

# 4d: Geographic distribution (latitude bins)
ax4 = fig.add_subplot(gs[1, :])
lat_bins = np.arange(-90, 100, 10)
df_meta['lat_bin'] = pd.cut(df_meta['latitude'], bins=lat_bins)

lat_dist = df_meta.groupby(['lat_bin', 'included']).size().unstack(fill_value=0)
lat_dist.plot(kind='bar', stacked=False, ax=ax4, color=['red', 'green'], alpha=0.7, width=0.8)
ax4.set_title('ROI Distribution by Latitude (10° bins)', fontweight='bold')
ax4.set_xlabel('Latitude Range')
ax4.set_ylabel('Number of ROIs')
ax4.legend(['Excluded', 'Included'], loc='upper right')
ax4.grid(axis='y', alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 4e: Detailed category breakdown
ax5 = fig.add_subplot(gs[2, :])
categories = ['Sea Only', 'Coastline Only', 'Coastline + Sea', 'Purely Land (Excluded)']
counts = [
    len(sea_roi_ids - coastline_roi_ids),
    len(coastline_roi_ids - sea_roi_ids),
    len(coastline_roi_ids & sea_roi_ids),
    len(purely_land_rois)
]
colors_bar = ['navy', 'orange', 'cyan', 'brown']
bars = ax5.barh(categories, counts, color=colors_bar, alpha=0.7, edgecolor='black')
ax5.set_xlabel('Number of ROIs')
ax5.set_title('Detailed Category Breakdown', fontweight='bold')
ax5.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax5.text(width, bar.get_y() + bar.get_height()/2.,
            f' {int(width):,} ({width/len(df_meta)*100:.1f}%)',
            ha='left', va='center', fontsize=10, fontweight='bold')

plt.savefig(f'{output_dir}/04_statistics.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 04_statistics.png")
plt.close()

# === Visualization 5: Coastline Buffer Visualization ===
print("\n8. Creating buffer visualization...")
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# Choose a sample region (Mediterranean)
sample_region = (-10, 40, 30, 46)
lon_min, lon_max, lat_min, lat_max = sample_region

for idx, (title, show_buffer) in enumerate([
    ('Original Coastline', False),
    ('With 300km Buffer', True),
    ('ROIs within Buffer', True)
]):
    ax = axes[idx]
    
    # Plot land
    gdf_land.cx[lon_min:lon_max, lat_min:lat_max].plot(
        ax=ax, color='wheat', edgecolor='darkgray', linewidth=0.5, alpha=0.7
    )
    
    # Plot coastline
    coastline_region = gdf_coastline.cx[lon_min:lon_max, lat_min:lat_max]
    coastline_region.plot(ax=ax, color='blue', linewidth=2, alpha=0.8, label='Coastline')
    
    # Plot buffer if requested
    if show_buffer:
        buffered_region = coastline_region.copy()
        buffered_region['geometry'] = buffered_region.geometry.buffer(buffer_distance)
        buffered_region.plot(ax=ax, color='lightblue', alpha=0.3, label='300km Buffer')
    
    # Plot ROIs for last subplot
    if idx == 2:
        region_rois = df_meta[
            (df_meta['longitude'] >= lon_min) & (df_meta['longitude'] <= lon_max) &
            (df_meta['latitude'] >= lat_min) & (df_meta['latitude'] <= lat_max)
        ]
        coastline_in_region = region_rois[region_rois['category'].isin(['Coastline Only', 'Coastline + Sea'])]
        not_coastline_in_region = region_rois[~region_rois['category'].isin(['Coastline Only', 'Coastline + Sea'])]
        
        ax.scatter(not_coastline_in_region['longitude'], not_coastline_in_region['latitude'],
                  c='gray', marker='x', s=30, alpha=0.5, label='Outside Buffer')
        ax.scatter(coastline_in_region['longitude'], coastline_in_region['latitude'],
                  c='red', marker='o', s=50, alpha=0.8, label='Within Buffer')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

plt.tight_layout()
plt.savefig(f'{output_dir}/05_buffer_visualization.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 05_buffer_visualization.png")
plt.close()

# === Visualization 6: Excluded ROIs Only ===
print("\n9. Creating excluded ROIs only visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Left: Global excluded ROIs
gdf_land.plot(ax=ax1, color='wheat', edgecolor='darkgray', linewidth=0.3, alpha=0.7)
gdf_coastline.plot(ax=ax1, color='blue', linewidth=0.5, alpha=0.5)

ax1.scatter(excluded_rois['longitude'], excluded_rois['latitude'], 
           c='darkred', marker='x', s=50, alpha=0.9, label=f'Excluded (Purely Land): {len(excluded_rois):,}')

ax1.set_title('Excluded ROIs Only - Global Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True, alpha=0.3)

# Right: Continental USA zoom showing excluded interior ROIs
sample_region = (-125, -65, 25, 50)
lon_min, lon_max, lat_min, lat_max = sample_region

gdf_land.cx[lon_min:lon_max, lat_min:lat_max].plot(
    ax=ax2, color='wheat', edgecolor='darkgray', linewidth=0.5, alpha=0.7
)
coastline_region = gdf_coastline.cx[lon_min:lon_max, lat_min:lat_max]
coastline_region.plot(ax=ax2, color='blue', linewidth=1.5, alpha=0.8, label='Coastline')

# Show 300km buffer zone
buffered_region = coastline_region.copy()
buffered_region['geometry'] = buffered_region.geometry.buffer(buffer_distance)
buffered_region.plot(ax=ax2, color='lightblue', alpha=0.2, label='300km Buffer Zone')

region_excluded = excluded_rois[
    (excluded_rois['longitude'] >= lon_min) & (excluded_rois['longitude'] <= lon_max) &
    (excluded_rois['latitude'] >= lat_min) & (excluded_rois['latitude'] <= lat_max)
]

ax2.scatter(region_excluded['longitude'], region_excluded['latitude'],
           c='red', marker='x', s=60, alpha=0.9, label=f'Excluded Interior ROIs: {len(region_excluded)}')

ax2.set_title('Continental USA: Excluded Interior ROIs', fontsize=14, fontweight='bold')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(lon_min, lon_max)
ax2.set_ylim(lat_min, lat_max)

plt.tight_layout()
plt.savefig(f'{output_dir}/06_excluded_only.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 06_excluded_only.png")
plt.close()

# === Summary ===
print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print(f"\nGenerated visualizations:")
print(f"  1. 01_global_overview.png       - Global map with all categories")
print(f"  2. 02_regional_views.png        - Zoom-ins of 4 regions")
print(f"  3. 03_density_heatmap.png       - Hexbin density maps")
print(f"  4. 04_statistics.png            - Statistical charts and distributions")
print(f"  5. 05_buffer_visualization.png  - Coastline buffer demonstration")
print(f"  6. 06_excluded_only.png         - Excluded ROIs only visualization")
print("\n" + "="*80)
