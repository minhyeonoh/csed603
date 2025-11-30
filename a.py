
import functools
import numpy as np
import typer
import vvv
import vvv.algorithm
import cv2

from typing_extensions import Annotated
from PIL import Image


app = typer.Typer()


@app.command()
def run(
  task_id: Annotated[
    str,
    typer.Argument(
      help="The ID of the task to run"
    )
  ],
  n_rounds: Annotated[
    int,
    typer.Option(
      help="The number of rounds to run"
    )
  ] = 100,
  n_posterior_samples: Annotated[
    int,
    typer.Option(
      help="The number of metropolis hastings samples from posterior"
    )
  ] = 1_000,
  burnin: Annotated[
    int,
    typer.Option(
      help="The number of burnin samples"
    )
  ] = 10_000,
  lag: Annotated[
    int,
    typer.Option(
      help="The number of iterations we run the sampler for between successive samples"
    )
  ] = 10,
  estimation_mode: Annotated[
    str,
    typer.Option(
      help="The estimation mode"
    )
  ] = "max",
  metropolis_hastings_init: Annotated[
    str,
    typer.Option(
      help="The initial value for metropolis hastings"
    )
  ] = "previous-posterior-mean",
  discount: Annotated[
    float,
    typer.Option(
      help="The discount"
    )
  ] = 0.0,
):

  # observation = Image.open("img.jpg")
  observation = Image.open("screenshot_pro/ex1.png")
  max_dim = max(observation.size)  # (width, height)
  if max_dim >= 500:
    target_max = 499
    scale = target_max / max_dim
    new_size = (max(1, int(observation.width * scale)),
          max(1, int(observation.height * scale)))
    observation = observation.resize(new_size, Image.LANCZOS)
  observation.save("highlighted.png")
  mask = np.ones(observation.size[::-1], dtype=np.uint8) * 255
  cv2.imwrite("mask.png", mask)
  observation = np.asarray(observation)
  algorithm = vvv.algorithm.EquiVolumeBisection(observation)
  algorithm.run()


@app.command()
def create():
  print("Creating user: Hiro Hamada")


@app.command()
def delete():
  print("Deleting user: Hiro Hamada")


if __name__ == "__main__":
  app()



