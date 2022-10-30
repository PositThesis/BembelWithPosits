
# copy geometry files for testing
configure_file(${CMAKE_SOURCE_DIR}/geo/sphere.dat ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/geo/torus.dat ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)

###############################################################################
# examples being run from CI
###############################################################################

set(CIFILES
    #AnsatzSpace
    #BlockClusterTree
    #Geometry
    LaplaceSingleLayerFull
    #LaplaceSingleLayerH2
    #HelmholtzSingleLayerFull
    #HelmholtzSingleLayerH2
    #MaxwellSingleLayerFull
    #MaxwellSingleLayerH2
    )

set(SCALARDEFINES
  ScalarIsPosit16
  ScalarIsPosit32
  ScalarIsPosit64
  ScalarIsFloat
  ScalarIsDouble
  ScalarIsLongDouble
)

foreach(scalar IN LISTS SCALARDEFINES)
  foreach(file IN LISTS CIFILES)
    add_executable(${file}${scalar} ${file})
    target_link_libraries(${file}${scalar} Eigen3::Eigen)
    target_compile_definitions(${file}${scalar} PUBLIC ${scalar})
    add_test(${file}${scalar}Example ${file}${scalar})
    install(TARGETS ${file}${scalar} DESTINATION bin)
  endforeach()
endforeach()

install(FILES ${CMAKE_SOURCE_DIR}/geo/sphere.dat DESTINATION bin)

#enable_testing()

###############################################################################
# examples not being run from CI
###############################################################################
#[[
set(NOCIFILES
    Quadrature
    VTKSurfaceExport
    VTKDomainExport
    FullLaplaceWorkflow
    )

foreach(file IN LISTS NOCIFILES)
  add_executable(${file}.out ${file})
  target_link_libraries(${file}.out Eigen3::Eigen)
endforeach()]]
