import folium
import numpy as np
import re
import csv

class Grid():
    def __init__(self, corner_bottom: tuple[float, float], corner_top: tuple[float, float], km_distance: float):
        self.corner_bottom = corner_bottom
        self.corner_top = corner_top
        self.height = corner_top[0] - corner_bottom[0]
        self.width = corner_top[1] - corner_bottom[1]
        self.degrees_between = km2deg(km_distance, corner_bottom[0])
        self.grid = [corner_bottom, corner_top]
        self.ratio = self.height / self.width

        if self.height < 0 or self.width < 0:
            raise ValueError("Points draw a negative box. \nInputs should be Grid(<left-bottom>, <top-right>, <distance>, <national>)")

        self.grid = self._make_grid()

    def _make_grid(self):
        grid = []
        lat_start, lon_start = self.corner_bottom
        lat_end, lon_end = self.corner_top

        lon = lon_start
        while lon <= lon_end:
            lat = lat_start
            while lat <= lat_end:
                grid.append((lat, lon))
                lat += self.degrees_between[1] * self.ratio
            lon += self.degrees_between[0]
        return grid

    def get_coords(self) -> list:
        return self.grid
    
    def show_grid(self):
        x, y = self.corner_bottom
        m = folium.Map(location=(x, y))
        self._draw_grid(m, self.grid)
        return m

    def _draw_grid(self, m, grid):
        for idx, point in enumerate(grid, start=1):
            x, y = point
            folium.Marker(
            location=[x, y],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background-color: red;
                    border-radius: 50%;
                    color: white;
                    width: 24px;
                    height: 24px;
                    text-align: center;
                    line-height: 24px;
                    font-size: 12px;
                    font-weight: bold;
                    border: 2px solid white;
                ">
                    {idx}
                </div>
                """
            )
            ).add_to(m)
    
    def plot_grid_from_list(self, areas):
        coords = self.grid
        m = folium.Map(location=self.corner_bottom, zoom_start=10)

        # Type colors and offsets (lat_offset, lon_offset)
        type_styles = {
            'wind': {'color': 'blue', 'offset': (-0.035, 0.08)},
            'sun': {'color': 'green', 'offset': (-0.035, -0.08)},
            'temp': {'color': 'purple', 'offset': (0.035, 0)},
        }

        for area in areas:
            if area == 'price':
                continue

            match_num = re.search(r'\d+', area)
            match_type = re.match(r'(wind|sun|temp)', area)

            if match_num:
                number = int(match_num.group())
                if 0 < number <= len(coords):
                    base_coord = coords[number - 1]
                    type_key = match_type.group() if match_type else ''
                    style = type_styles.get(type_key, {'color': 'red', 'offset': (0, 0)})

                    # Apply offset
                    lat_offset, lon_offset = style['offset']
                    coord = (base_coord[0] + lat_offset, base_coord[1] + lon_offset)

                    folium.Marker(
                        location=coord,
                        icon=folium.DivIcon(
                            html=f"""
                            <div style="
                                background-color: {style['color']};
                                border-radius: 50%;
                                color: white;
                                width: 24px;
                                height: 24px;
                                text-align: center;
                                line-height: 24px;
                                font-size: 12px;
                                font-weight: bold;
                                border: 2px solid white;
                            ">
                                {number}
                            </div>
                            """
                        ),
                        popup=area
                    ).add_to(m)
                else:
                    print(f"Index {number} out of range for grid")
            else:
                print(f"No index found in area name: {area}")
        print("PLOT LEGEND")
        for typ, val in type_styles.items():
            color = val['color']
            print(f"{color}: {typ}")
            
        return m
    
    def get_area_list_from_csv(self, file_path):

        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) 
            area_list = [row[0] for row in reader]

        return area_list




def km2deg(km, lattitude):
    degree = 111.32
    lat = km / degree
    lon = km / (degree * np.cos(np.radians(lattitude)))
    return (float(lat), float(lon))