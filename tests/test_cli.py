from riskaware_eda.cli import main


def test_small_demo_cli(tmp_path):
    output = tmp_path / "demo"
    return_code = main(
        [
            "demo",
            "--output",
            str(output),
            "--circuits",
            "6",
            "--recipes",
            "20",
            "--length",
            "5",
            "--budget",
            "6",
            "--trees",
            "20",
            "--alpha",
            "0.1",
        ]
    )
    assert return_code == 0
    assert (output / "risk_model.joblib").is_file()
    assert (output / "simulation.json").is_file()
