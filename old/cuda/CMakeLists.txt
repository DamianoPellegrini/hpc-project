add_executable(cuda_app main.cu)
target_link_libraries(cuda_app PRIVATE mst_core)
target_compile_features(cuda_app PRIVATE cxx_std_20)

set_target_properties(
  cuda_app
  PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
