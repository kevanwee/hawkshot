import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math

def line_of_sight_with_elevation_profile(dem_array, coord_a, coord_b, 
                                        observer_height, target_height, 
                                        geo_transform=None, sample_spacing):
    """
    check if point B is visible from point A.
    
    parameters:
        coord_a: (row, col) of observer
        coord_b: (row, col) of target
        observer_height: height above ground in meters
        target_height: height above ground in meters
        geo_transform: GDAL geotransform for converting to real distances
    """
    # Extract coordinates
    row_a, col_a = coord_a
    row_b, col_b = coord_b
    
    # Get base elevations and add heights
    observer_base_elev = dem_array[row_a, col_a]
    target_base_elev = dem_array[row_b, col_b]
    observer_elev = observer_base_elev + observer_height
    target_elev = target_base_elev + target_height
    
    # Calculate total distance in meters using geotransform
    if geo_transform is not None:
        # Calculate distance in real-world coordinates
        pixel_width = abs(geo_transform[1])  # meters per pixel in x
        pixel_height = abs(geo_transform[5])  # meters per pixel in y
        
        # Convert pixel distances to real-world distances using approximate conversion
        # For lat/lon data, we need to convert degrees to approximate meters
        lon_diff = abs(col_b - col_a) * pixel_width
        lat_diff = abs(row_b - row_a) * pixel_height
        
        # Approximate conversion from degrees to kilometers (1° lat ≈ 111 km)
        # More accurate calculation for real-world distances in km
        lat_avg = (row_a + row_b) / 2 * pixel_height + geo_transform[3]
        lon_km_per_deg = 111 * math.cos(math.radians(abs(lat_avg)))
        
        dx_km = lon_diff * lon_km_per_deg
        dy_km = lat_diff * 111
        
        total_distance = math.sqrt(dx_km**2 + dy_km**2) * 1000  # Convert to meters
    else:
        # Fallback to pixel distance if no geotransform
        total_distance = math.sqrt((col_b - col_a)**2 + (row_b - row_a)**2)
    
    # Calculate number of samples based on pixel distance and sampling density
    pixel_distance = math.sqrt((col_b - col_a)**2 + (row_b - row_a)**2)
    num_steps = max(100, int(pixel_distance / sample_spacing))
    
    print(f"Total distance: {total_distance:.1f}m, Number of samples: {num_steps}")
    
    # Create distance array with specified spacing
    distances = np.linspace(0, total_distance, num_steps)
    
    # Generate intermediate points along the line
    rows = np.linspace(row_a, row_b, num_steps)
    cols = np.linspace(col_a, col_b, num_steps)
    
    # Arrays to store elevation profile data
    terrain_profile = np.zeros(num_steps)
    sight_line_profile = np.zeros(num_steps)
    
    # Set observer and target in profiles
    terrain_profile[0] = observer_base_elev
    terrain_profile[-1] = target_base_elev
    sight_line_profile[0] = observer_elev
    sight_line_profile[-1] = target_elev
    
    # Check line of sight and build elevation profile
    obstruction_idx = None
    
    for i in range(1, num_steps-1):
        # Get the nearest pixel for each interpolated point
        row_idx = int(round(rows[i]))
        col_idx = int(round(cols[i]))
        
        # Ensure within bounds
        if (0 <= row_idx < dem_array.shape[0] and 
            0 <= col_idx < dem_array.shape[1]):
            
            # Get terrain elevation
            terrain_elev = dem_array[row_idx, col_idx]
            terrain_profile[i] = terrain_elev
            
            # Calculate precise distance ratio
            dist_ratio = distances[i] / distances[-1]
            
            # Calculate expected sight line elevation at this point
            sight_line = observer_elev + (target_elev - observer_elev) * dist_ratio
            sight_line_profile[i] = sight_line
            
            # Apply earth curvature correction for long distances
            if distances[-1] > 5000:  # If > 5km
                dist_km = distances[i] / 1000
                curvature_drop = (dist_km**2) / 12.74  # Approximation in meters
                adjusted_sight_line = sight_line - curvature_drop
                sight_line_profile[i] = adjusted_sight_line
            else:
                adjusted_sight_line = sight_line
            
            # Check if terrain blocks sight line
            if terrain_elev > adjusted_sight_line and obstruction_idx is None:
                obstruction_idx = i
                obstruction_point = (row_idx, col_idx)
                print(f"Obstruction found at {distances[i]:.1f}m: Terrain={terrain_elev:.1f}m, Sight line={adjusted_sight_line:.1f}m")
    
    # Determine visibility
    visible = obstruction_idx is None
    
    # Prepare result data
    profile_data = {
        'distances': distances,
        'terrain': terrain_profile,
        'sight_line': sight_line_profile,
        'obstruction_idx': obstruction_idx
    }
    
    return visible, obstruction_point if not visible else None, profile_data

def viewshed_with_profile(dem_path, observer_coords, target_coords, 
                         observer_height, target_height, sample_spacing):
    """
    Visualize line of sight between two coordinates with elevation profile.
    
    Parameters:
        dem_path: Path to the DEM file
        observer_coords: (longitude, latitude) of observer
        target_coords: (longitude, latitude) of target
        observer_height: Height above ground in meters
        target_height: Height above ground in meters
    """
    # Open and process DEM
    dem_ds = gdal.Open(dem_path)
    if dem_ds is None:
        raise FileNotFoundError(f"DEM file not found at path: {dem_path}")
    
    dem_band = dem_ds.GetRasterBand(1)
    dem_array = dem_band.ReadAsArray()
    geo_transform = dem_ds.GetGeoTransform()
    
    # Convert coordinates to pixel indices
    def coord_to_pixel(lon, lat):
        x = int((lon - geo_transform[0]) / geo_transform[1])
        y = int((lat - geo_transform[3]) / geo_transform[5])
        return x, y
    
    observer_x, observer_y = coord_to_pixel(*observer_coords)
    target_x, target_y = coord_to_pixel(*target_coords)
    
    print(f"Observer pixel coordinates: ({observer_x}, {observer_y})")
    print(f"Target pixel coordinates: ({target_x}, {target_y})")
    
    # Perform line of sight analysis with profile data
    visible, obstruction_point, profile_data = line_of_sight_with_elevation_profile(
        dem_array, 
        (observer_y, observer_x), 
        (target_y, target_x),
        observer_height,
        target_height,
        geo_transform,
        sample_spacing
    )
    
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # DEM view subplot
    ax1 = fig.add_subplot(gs[0])
    dem_plot = ax1.imshow(dem_array, cmap='terrain')
    

    ax1.plot(observer_x, observer_y, 'g*', markersize=15, 
            label=f'Observer (h={observer_height}m)')
    ax1.plot(target_x, target_y, 'r^', markersize=15, 
           label=f'Target (h={target_height}m)')

    if visible:
        ax1.plot([observer_x, target_x], [observer_y, target_y], 
                'g-', linewidth=2, label='Clear Line of Sight')
        ax1.set_title('Target is Visible', fontsize=16, weight='bold')
    else:
        obst_x, obst_y = obstruction_point[1], obstruction_point[0]
        ax1.plot([observer_x, obst_x], [observer_y, obst_y], 
                'g-', linewidth=2, label='Visible Segment')
        ax1.plot([obst_x, target_x], [obst_y, target_y], 
                'r--', linewidth=2, label='Obstructed Segment')
        ax1.plot(obst_x, obst_y, 'yo', markersize=12, 
                label='First Obstruction')
        ax1.set_title('Target is NOT Visible', fontsize=16, weight='bold')
    
    # add colorbar and labels
    plt.colorbar(dem_plot, ax=ax1, label='Elevation (m)', shrink=0.7)
    ax1.set_xlabel('Column Pixel', fontsize=12)
    ax1.set_ylabel('Row Pixel', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # elevation profile subplot
    ax2 = fig.add_subplot(gs[1])
    
    scaled_distances = profile_data['distances'] / 1000  # Convert to kilometers
    
    # plot terrain
    ax2.fill_between(scaled_distances, 0, profile_data['terrain'], 
                    alpha=0.5, color='green', label='Terrain')
    
    # plot los
    ax2.plot(scaled_distances, profile_data['sight_line'], 
            'b-', linewidth=2, label='Line of Sight')
    
    # show obstruction
    if not visible and profile_data['obstruction_idx'] is not None:
        obst_idx = profile_data['obstruction_idx']
        obst_dist = scaled_distances[obst_idx]
        obst_elev = profile_data['terrain'][obst_idx]
        sight_line_elev = profile_data['sight_line'][obst_idx]
        
        ax2.plot(obst_dist, obst_elev, 'ro', markersize=10, 
                label='Obstruction Point')
        
        # Annotate the obstruction
        ax2.annotate(f'Obstruction: {obst_elev:.1f}m\nSight line: {sight_line_elev:.1f}m',
                    xy=(obst_dist, obst_elev),
                    xytext=(obst_dist + max(scaled_distances)/20, 
                           obst_elev + (max(profile_data['terrain']) - min(profile_data['terrain']))/10),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2))
    
    # add observer and target markers on profile
    ax2.plot(scaled_distances[0], profile_data['sight_line'][0], 'g*', 
            markersize=12, label='Observer')
    ax2.plot(scaled_distances[-1], profile_data['sight_line'][-1], 'r^', 
            markersize=12, label='Target')
    
    # set y-axis to start at 0
    ax2.set_ylim(bottom=0)
    
    # profile labels and legend
    ax2.set_title('Elevation Profile', fontsize=14)
    ax2.set_xlabel('Distance (kilometers)', fontsize=12)  # display in kilometers
    ax2.set_ylabel('Elevation (meters)', fontsize=12)
    ax2.grid(True)
    ax2.legend(loc='upper right')
    
    # Tight layout and show
    plt.tight_layout()
    plt.show()
    
    return dem_array, visible, obstruction_point, profile_data

# Example Usage
if __name__ == "__main__":
    dem_path = 'dted.dt2'
    observer_coords = (-115.68148,42.93113)  # Longitude, Latitude
    target_coords = (-115.71633,42.85085)    # Longitude, Latitude
    
    # Height parameters (in meters)
    observer_height = 0   # Observer at ground level
    target_height = 0   # Target with 1000m height

    # Keep the 0.001 sample spacing for accurate DTED representation
    sample_spacing = 5 # Adjust for less detail if needed  
    
    try:
        dem_array, visible, obstruction, profile = viewshed_with_profile(
            dem_path,
            observer_coords,
            target_coords,
            observer_height,
            target_height,
            sample_spacing
        )
        

        observer_base = profile['terrain'][0]
        target_base = profile['terrain'][-1]
        
        print("\n===== VISIBILITY ANALYSIS RESULTS =====")
        print(f"Observer base elevation: {observer_base:.1f}m + {observer_height}m height")
        print(f"Target base elevation: {target_base:.1f}m + {target_height}m height")
        print(f"Total samples along path: {len(profile['distances'])}")
        print(f"Total distance: {profile['distances'][-1]/1000:.2f} km")
        
        if visible:
            print("\n✅ TARGET IS VISIBLE")
            print(f"   Observer effective height: {observer_base + observer_height:.1f}m")
            print(f"   Target effective height: {target_base + target_height:.1f}m")
            print(f"   Maximum terrain elevation along path: {max(profile['terrain']):.1f}m")
        else:
            print("\n❌ TARGET IS NOT VISIBLE")
            obstruction_idx = profile['obstruction_idx']
            print(f"   Obstruction at distance: {profile['distances'][obstruction_idx]/1000:.2f} km")
            print(f"   Obstruction elevation: {profile['terrain'][obstruction_idx]:.1f}m")
            print(f"   Sight line at obstruction: {profile['sight_line'][obstruction_idx]:.1f}m")
            print(f"   Height needed to clear: {profile['terrain'][obstruction_idx] - profile['sight_line'][obstruction_idx] + 1:.1f}m")
            
    except Exception as e:
        print(f"🚨 Error: {e}")
