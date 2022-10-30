add_executable(test_Spline.out test_Spline)
target_link_libraries(test_Spline.out Eigen3::Eigen Bembel::Bembel)
target_include_directories(test_Spline.out PUBLIC universal)
add_test(Spline test_Spline.out)

add_executable(test_GeometryImportAndEval.out test_GeometryImportAndEval)
target_link_libraries(test_GeometryImportAndEval.out Eigen3::Eigen Bembel::Bembel)
target_include_directories(test_GeometryImportAndEval.out PUBLIC universal)
add_test(GeometryImportAndEval test_GeometryImportAndEval.out)

add_executable(test_SurfacePointUpdate.out test_SurfacePointUpdate)
target_link_libraries(test_SurfacePointUpdate.out Eigen3::Eigen Bembel::Bembel)
target_include_directories(test_SurfacePointUpdate.out PUBLIC universal)
add_test(SurfacePointUpdate test_SurfacePointUpdate.out)

add_executable(test_Projector.out test_Projector)
target_link_libraries(test_Projector.out Eigen3::Eigen Bembel::Bembel)
target_include_directories(test_Projector.out PUBLIC universal)
add_test(Projector test_Projector.out)

add_executable(test_Glue.out test_Glue)
target_link_libraries(test_Glue.out Eigen3::Eigen Bembel::Bembel)
target_include_directories(test_Glue.out PUBLIC universal)
add_test(Glue test_Glue.out)

add_executable(test_FMMTransferMatrices.out test_FMMTransferMatrices)
target_link_libraries(test_FMMTransferMatrices.out Eigen3::Eigen Bembel::Bembel)
target_include_directories(test_FMMTransferMatrices.out PUBLIC universal)
add_test(FMMTransferMatrices test_FMMTransferMatrices.out)

add_executable(test_FMMForwardTransformation.out test_FMMForwardTransformation)
target_link_libraries(test_FMMForwardTransformation.out Eigen3::Eigen Bembel::Bembel)
target_include_directories(test_FMMForwardTransformation.out PUBLIC universal)
add_test(FMMForwardTransformation test_FMMForwardTransformation.out)

add_executable(test_FMMBackwardTransformation.out test_FMMBackwardTransformation)
target_link_libraries(test_FMMBackwardTransformation.out Eigen3::Eigen Bembel::Bembel)
target_include_directories(test_FMMBackwardTransformation.out PUBLIC universal)
add_test(FMMBackwardTransformation test_FMMBackwardTransformation.out)

add_executable(test_DuffyTrick.out test_DuffyTrick)
target_link_libraries(test_DuffyTrick.out Eigen3::Eigen Bembel::Bembel)
target_include_directories(test_DuffyTrick.out PUBLIC universal)
add_test(DuffyTrick test_DuffyTrick.out)

enable_testing()
