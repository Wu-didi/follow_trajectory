import csv
from pyproj import CRS, Transformer

def latlon_to_utm_zone(latitude, longitude):
    """
    根据经纬度计算所属的UTM投影带号及EPSG代码。
    """
    zone_number = int((longitude + 180) // 6) + 1
    if latitude >= 0:
        epsg_code = f"326{zone_number:02d}"
    else:
        epsg_code = f"327{zone_number:02d}"
    return zone_number, epsg_code

def latlon_to_utm(latitude, longitude):
    """
    将WGS84坐标(经纬度)转换为对应UTM坐标(Easting, Northing)。
    """
    _, utm_epsg = latlon_to_utm_zone(latitude, longitude)
    wgs84 = CRS.from_epsg(4326)
    utm = CRS.from_epsg(int(utm_epsg))
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)
    easting, northing = transformer.transform(longitude, latitude)
    return easting, northing

def convert_latlon_csv_to_utm_csv(input_csv, output_csv):
    with open(input_csv, mode='r', newline='') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['Easting', 'Northing', 'Heading', 'Speed']  # 输出列字段
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            lon = float(row['Longitude'])
            lat = float(row['Latitude'])
            heading = float(row['Heading'])
            speed = float(row['Speed'])

            # 转换为UTM坐标
            easting, northing = latlon_to_utm(lat, lon)

            out_row = {
                'Easting': easting,
                'Northing': northing,
                'Heading': heading,
                'Speed': speed
            }
            writer.writerow(out_row)



if __name__ == "__main__":
    input_file = "/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_shiyanzhongxin_0327.csv"
    output_file = "/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_shiyanzhongxin_0327_utm.csv"
    convert_latlon_csv_to_utm_csv(input_file, output_file)
    print("转换完成！")
