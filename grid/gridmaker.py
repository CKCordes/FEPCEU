import folium
import numpy as np

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
        self._draw_grid(m)
        return m

    def _draw_grid(self, m):
        for idx, point in enumerate(self.grid, start=1):
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


def km2deg(km, lattitude):
    degree = 111.32
    lat = km / degree
    lon = km / (degree * np.cos(np.radians(lattitude)))
    return (float(lat), float(lon))


