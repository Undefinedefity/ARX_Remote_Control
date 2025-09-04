import zarr
import numpy as np
from typing import Optional
from numcodecs import Blosc

def read_in_depth(zarr_path: str) -> np.ndarray:
    """
    从 Zarr 文件读取深度数据（固定从 "depth" 数据集读取）
    """
    try:
        zarr_group = zarr.open(zarr_path, mode='r')
        
        # 更安全的检查方式
        if not any(key == "depth" for key in zarr_group.keys()):
            raise KeyError("Zarr 文件中必须包含 'depth' 数据集")
            
        depth_data = zarr_group["depth"]
        
        #if depth_data.ndim != 2:
         #   raise ValueError(f"深度数据应为 2D 数组，但获取到 {depth_data.ndim}D 数据")
            
        return np.array(depth_data)
        
    except zarr.errors.PathNotFoundError:
        raise FileNotFoundError(f"Zarr 路径不存在: {zarr_path}")

def generate_pcd_zarr(
    pcd_array: np.ndarray,
    output_zarr_path: str,
    chunk_shape: tuple = (50,1024, 3),#1024
    compressor: Optional[Blosc] = None,
    dataset_name: str = "pointcloud"  # 修正为小写保持一致性
) -> None:
    """
    将点云保存到 Zarr 文件的指定数据集
    """
    #if pcd_array.ndim != 2 or pcd_array.shape[1] != 3:
     #  raise ValueError("点云必须是(N,3)形状的数组")

    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=5, shuffle=2)

    root = zarr.open_group(output_zarr_path, mode='w')
    root.array(
        name=dataset_name,
        data=pcd_array,
        chunks=chunk_shape,
        compressor=compressor
    )

#the following code is for testing
if __name__ == "__main__":
    # 测试 read_in_depth
    try:
        # 正确创建包含 "depth" 数据集的测试文件
        test_depth = np.random.rand(480, 640).astype(np.float32)
        root = zarr.open_group("test_depth.zarr", mode='w')
        root.array(name="depth", data=test_depth)
        
        # 测试读取
        depth_data = read_in_depth("test_depth.zarr")
        assert depth_data.shape == (480, 640), "形状验证失败"
        
        # 测试异常情况
        try:
            read_in_depth("invalid_path.zarr")
        except FileNotFoundError:
            pass
            
        print("[read_in_depth 测试] 全部通过")
        
    except Exception as e:
        print(f"read_in_depth 测试失败: {str(e)}")

    # 测试 generate_pcd_zarr
    try:
        test_pcd = np.random.rand(1000, 3).astype(np.float32)
        generate_pcd_zarr(test_pcd, "test_output.zarr")
        
        # 验证输出
        grp = zarr.open("test_output.zarr")
        assert "pointcloud" in grp, "数据集不存在"
        assert grp["pointcloud"].shape == (1000, 3), "形状不匹配"
        assert np.allclose(grp["pointcloud"][:], test_pcd), "数据不一致"
        
        print("[generate_pcd_zarr 测试] 全部通过")
        
    except Exception as e:
        print(f"generate_pcd_zarr 测试失败: {str(e)}")