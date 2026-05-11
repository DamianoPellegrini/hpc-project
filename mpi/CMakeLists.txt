add_executable(mpi_app main.cpp)
target_link_libraries(mpi_app PRIVATE mst_common MPI::MPI_CXX)
target_compile_features(mpi_app PRIVATE cxx_std_17)
