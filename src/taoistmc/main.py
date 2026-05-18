import typer
import shutil
from typing import Optional
from pathlib import Path
from importlib import resources # Modern way to access package data
from taoistmc import TaoistMc  # Adjust import based on your file structure

app = typer.Typer(help="taoistmc: Transmission of Absorbers in the Intergalactic Space Tool")

@app.command()
def init(
    output: Path = typer.Option(
        "config.yaml", 
        "--output", "-o", 
        help="The name of the config file to create"
    ),
    force: bool = typer.Option(
        False, 
        "--force", "-f", 
        help="Overwrite existing config.yaml if it exists"
    )
):
    """
    Copy a starter configuration file to the current directory.
    """
    # 1. Check if file already exists to avoid accidental overwrites
    if output.exists() and not force:
        typer.secho(
            f"❌ Error: {output} already exists. Use --force to overwrite.", 
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # 2. Locate the internal template using importlib.resources
    # 'taoistmc.data' is the package path, 'starter_config.yaml' is the file
    try:
        # For Python 3.9+:
        source_path = resources.files("taoistmc.data").joinpath("starter_config.yaml")
        
        with source_path.open("rb") as src, output.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            
        typer.secho(f"✅ Created starter config: {output} with parameters from Steidel et al. 2018", fg=typer.colors.GREEN, bold=True)
        typer.echo("Edit this file to customize your physics parameters before running 'run'.")
        
    except Exception as e:
        typer.secho(f"❌ Failed to copy starter config: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

@app.command()
def run(
    z_em: float = typer.Argument(..., help="The source redshift for the simulation"),
    n: int = typer.Option(100, "--nsightlines", "-n", help="Total number of sightlines requested"),
    config_path: Path = typer.Option(
        "config.yaml", 
        "--config", "-c", 
        help="Path to the YAML configuration file"
    ),
    verbose: Optional[bool] = typer.Option(None, "--verbose/--quiet", "-v/-q", help="Override config verbosity")
):
    """
    Generate IGM transmission curves for a given redshift.
    Checks for existing valid runs and only generates the delta.
    """
    # 1. Check if the config file exists
    if not config_path.exists():
        typer.secho(f"❌ Configuration file '{config_path}' not found.", fg=typer.colors.RED, bold=True)
        typer.echo("\nTo get started, generate a starter config by running:")
        typer.secho(f"    taoistmc init", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"\nThen edit '{config_path}' to match your physics parameters.")
        raise typer.Exit(code=1)

    # 2. Proceed with initialization if it exists
    try:
        mc = TaoistMc.from_yaml(str(config_path))
    except Exception as e:
        typer.secho(f"❌ Error parsing '{config_path}':", fg=typer.colors.RED, bold=True)
        typer.echo(str(e))
        raise typer.Exit(code=1)
    
    # 3. Manual overrides from CLI
    if verbose is not None:
        mc.config.verbose = verbose

    typer.echo(f"🚀 Initializing TAOIST for z = {z_em:.3f} (Target: {n} sightlines)")

    # 4. Run the simulation (This will handle loading and N-M logic)
    results = mc.run(z_em=z_em, n_sightlines=n)
    
    total_count = len(results)
    
    # 5. Final summary
    typer.secho(f"✅ Simulation Complete!", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Total valid sightlines now available: {total_count}")
    
    if mc.config.save:
        z_str = f"z{str(z_em).replace('.', 'p')}"
        save_loc = Path(mc.config.output_dir) / z_str
        typer.echo(f"Data stored in: {save_loc}")

if __name__ == "__main__":
    app()