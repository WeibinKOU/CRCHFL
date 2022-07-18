import open3d as o3d
import argparse

def main():
    argparser = argparse.ArgumentParser(description='Visualize points cloud')
    argparser.add_argument('--file', type=str, help='to specify the points cloutd file path')
    args = argparser.parse_args()

    pcd = o3d.io.read_point_cloud(args.file)
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
