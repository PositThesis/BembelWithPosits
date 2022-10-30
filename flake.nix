{
  description = "A very basic flake";

  inputs = {
    bembel = {
      url = "github:PositThesis/BembelTemplating";
    };
    online_lib = {
        url = "github:PositThesis/EigenUniversalIntegration";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, bembel, online_lib, flake-utils }:
    let
      # wrap this in another let .. in to add the hydra job only for a single architecture
      output_set = flake-utils.lib.eachDefaultSystem (system:
        let
            pkgs = nixpkgs.legacyPackages.${system};
            bembel_nocheck = bembel.packages.${system}.bembel.overrideAttrs (oldAttrs: rec { doCheck = false; });
            eigen_with_dgb = online_lib.packages.${system}.eigen.overrideAttrs (oldAttrs: rec{
                postPatch = ''
                    sed -i 's/m_eivalues \*= scale;/std::cout << "rescaling " << m_eivalues.size() << std::endl; m_eivalues *= scale; std::cout << "rescale done" << std::endl;/g' Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h
                    grep m_eivalues Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h
                    echo "#include <iostream>" | cat - Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h > temp
                    mv temp Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h
                '';
            });
        in
        rec {
            packages = flake-utils.lib.flattenTree {
                bembel_posits = pkgs.gcc10Stdenv.mkDerivation {
                    name = "Bembel with Posits";
                    src = ./.;

                    nativeBuildInputs = [pkgs.cmake];

                    buildInputs = [
                        online_lib.packages.${system}.eigen
                        # eigen_with_dgb
                        online_lib.packages.${system}.universal
                        online_lib.packages.${system}.eigen_universal_integration
                        # bembel.packages.${system}.bembel
                        bembel_nocheck
                        pkgs.llvmPackages.openmp
                    ];

                    checkPhase = ''
                        ctest
                    '';

                    doCheck = false;
                };
            };

            devShell = pkgs.mkShell {
                buildInputs = [
                    online_lib.packages.${system}.eigen
                    online_lib.packages.${system}.universal
                    online_lib.packages.${system}.eigen_universal_integration
                    # bembel.packages.${system}.bembel
                    bembel_nocheck
                    pkgs.llvmPackages.openmp
                    pkgs.cmake
                ];

                shellHook = ''
                    function run_cmake_build() {
                        cmake -B /mnt/RamDisk/bempos_build -DCMAKE_BUILD_TYPE=Debug
                        cmake --build /mnt/RamDisk/bempos_build -j6 --target all
                    }
                '';
            };

            defaultPackage = packages.bembel_posits;
        }
    );
    in
      output_set // { hydraJobs.build."aarch64-linux" = output_set.defaultPackage."aarch64-linux"; };
    }
