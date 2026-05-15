add_executable(openmp_app main.cpp)
target_link_libraries(openmp_app PRIVATE mst_core OpenMP::OpenMP_CXX)
target_compile_features(openmp_app PRIVATE cxx_std_20)
