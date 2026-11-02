from scripts.openevo_runtime.production_package import build_production_package, validate_production_package


def test_production_package_is_exact_and_non_authoritative() -> None:
    package = build_production_package()
    validate_production_package(package)
    assert len(package["package_identity"]) == 64
    assert package["python_version"] == "3.13.3"
    assert package["temporalio_version"] == "1.33.0"
    assert package["scientific_authority"] is False
    assert package["formal_launch_authority"] is False
    assert package["gpu_authority"] is False
    assert package["final_panel_authority"] is False
    paths = {row["path"] for row in package["files"]}
    assert "scripts/openevo_runtime/temporal_zero_formal.py" in paths
    assert "scripts/openevo_runtime/resource_broker.py" in paths
    assert "requirements-execution-runtime-202609171852.txt" in paths
    assert "scripts/openevo_webshop/opportunistic_canary_guard.py" not in paths
    assert "scripts/openevo_webshop/formal_priority_witness.py" not in paths


def test_production_package_identity_is_deterministic() -> None:
    assert build_production_package()["package_identity"] == build_production_package()["package_identity"]


def test_production_package_closes_repo_local_runtime_dependencies() -> None:
    from scripts.openevo_runtime import production_package as package_mod

    package = package_mod.build_production_package()
    paths = {row["path"] for row in package["files"]}
    required = {
        "scripts/openevo_webshop/elastic_gpu_lane.py",
        "schemas/experiment-passport-202609111800.schema.json",
        "schemas/experiment-registry-202609112120.schema.json",
        "schemas/experiment-passport-operation-node-202609202238.schema.json",
        "schemas/experiment-passport-operation-node-202609210003.schema.json",
        "schemas/experiment-passport-artifact-edge-202609202238.schema.json",
        "schemas/experiment-passport-artifact-index-202609202238.schema.json",
    }
    assert required <= paths
    backend_deps = package_mod._static_python_dependencies(
        "scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py"
    )
    assert "scripts/openevo_webshop/elastic_gpu_lane.py" in backend_deps
    dag_deps = package_mod._static_python_dependencies(
        "scripts/openevo_webshop/experiment_artifact_dag.py"
    )
    assert "schemas/experiment-passport-operation-node-202609210003.schema.json" in dag_deps
