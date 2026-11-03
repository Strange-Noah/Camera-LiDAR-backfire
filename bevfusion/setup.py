import os
from pathlib import Path
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def make_cuda_ext(name, module, sources, sources_cuda=None, extra_args=None, extra_include_path=None):
    sources_cuda = sources_cuda or []
    extra_args = extra_args or []
    extra_include_path = extra_include_path or []
    define_macros = []
    compile_args = {"cxx": list(extra_args)}
    use_cuda = ((torch.cuda.is_available() and torch.version.cuda is not None)
                or os.getenv("FORCE_CUDA", "0") == "1")
    use_rocm = ((torch.cuda.is_available() and torch.version.hip is not None)
                or os.getenv("FORCE_ROCM", "0") == "1")
    if use_cuda or use_rocm:
        define_macros.append(("WITH_CUDA" if use_cuda else "WITH_ROCM", None))
        extension = CUDAExtension
        compile_args["nvcc" if use_cuda else "hipcc"] = list(extra_args) + [
            "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        # PyTorch derives gencode flags from TORCH_CUDA_ARCH_LIST.
        sources = list(sources) + list(sources_cuda)
    else:
        print(f"Compiling {name} without CUDA")
        extension = CppExtension
    return extension(
        name=f"{module}.{name}",
        sources=[os.path.join(*module.split("."), src) for src in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=compile_args,
    )


if __name__ == "__main__":
    setup(
        name="mmdet3d", packages=find_packages(), include_package_data=True,
        package_data={"mmdet3d.ops": ["*/*.so"]},
        ext_modules=[
            make_cuda_ext("sparse_conv_ext", "mmdet3d.ops.spconv",
                ["src/all.cc", "src/reordering_cpu.cc", "src/reordering_cuda.cu",
                 "src/indice_cpu.cc", "src/indice_cuda.cu", "src/maxpool_cpu.cc",
                 "src/maxpool_cuda.cu"], extra_args=["-w", "-std=c++17"],
                extra_include_path=[str(Path("mmdet3d/ops/spconv/include").resolve())]),
            make_cuda_ext("bev_pool_ext", "mmdet3d.ops.bev_pool",
                ["src/bev_pool_cpu.cpp"], ["src/bev_pool_cuda.cu"]),
            make_cuda_ext("iou3d_cuda", "mmdet3d.ops.iou3d",
                ["src/iou3d.cpp"], ["src/iou3d_kernel.cu"]),
            make_cuda_ext("voxel_layer", "mmdet3d.ops.voxel",
                ["src/voxelization.cpp", "src/scatter_points_cpu.cpp",
                 "src/voxelization_cpu.cpp"],
                ["src/scatter_points_cuda.cu", "src/voxelization_cuda.cu"]),
            make_cuda_ext("roiaware_pool3d_ext", "mmdet3d.ops.roiaware_pool3d",
                ["src/roiaware_pool3d.cpp", "src/points_in_boxes_cpu.cpp"],
                ["src/roiaware_pool3d_kernel.cu", "src/points_in_boxes_cuda.cu"]),
            make_cuda_ext("ball_query_ext", "mmdet3d.ops.ball_query",
                ["src/ball_query_cpu.cpp"], ["src/ball_query_cuda.cu"]),
            make_cuda_ext("knn_ext", "mmdet3d.ops.knn",
                ["src/knn_cpu.cpp"], ["src/knn_cuda.cu"]),
            make_cuda_ext("assign_score_withk_ext", "mmdet3d.ops.paconv",
                ["src/assign_score_withk.cpp"], ["src/assign_score_withk_cuda.cu"]),
            make_cuda_ext("group_points_ext", "mmdet3d.ops.group_points",
                ["src/group_points_cpu.cpp"], ["src/group_points_cuda.cu"]),
            make_cuda_ext("interpolate_ext", "mmdet3d.ops.interpolate",
                ["src/interpolate.cpp"],
                ["src/three_interpolate_cuda.cu", "src/three_nn_cuda.cu"]),
            make_cuda_ext("furthest_point_sample_ext", "mmdet3d.ops.furthest_point_sample",
                ["src/furthest_point_sample_cpu.cpp"],
                ["src/furthest_point_sample_cuda.cu"]),
            make_cuda_ext("gather_points_ext", "mmdet3d.ops.gather_points",
                ["src/gather_points_cpu.cpp"], ["src/gather_points_cuda.cu"]),
        ],
        cmdclass={"build_ext": BuildExtension}, zip_safe=False,
    )