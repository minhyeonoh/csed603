
import abc
import base64
import numpy as np
import vvv.random
import litellm

from pydantic import BaseModel
from PIL import Image


def image_to_base64(image_path):
  """
  Encodes an image file to a Base64 string.

  Args:
      image_path (str): The path to the image file.

  Returns:
      str: The Base64 encoded string of the image.
  """
  try:
    with open(image_path, "rb") as image_file:
      # Read the image content as bytes
      image_bytes = image_file.read()
      # Encode the bytes to Base64
      encoded_image_bytes = base64.b64encode(image_bytes)
      # Decode the Base64 bytes to a UTF-8 string
      encoded_image_string = encoded_image_bytes.decode('utf-8')
      return encoded_image_string
  except FileNotFoundError:
    return "Error: Image file not found at " + image_path
  except Exception as e:
    return "An error occurred during encoding: " + str(e)


def get_query(observation, cut):

  h, w, *_ = observation.shape
  xx, yy = np.meshgrid(np.arange(w), np.arange(h))
  coords = np.column_stack([xx.flatten(), yy.flatten()])
  dot = np.dot(coords.astype(float) - cut[np.newaxis, ...], cut)

  observation_p = np.array(observation)
  at = coords[dot < 0]
  observation_p[at[:, 1], at[:, 0]] = 0

  observation_n = np.array(observation)
  at = coords[dot >= 0]
  observation_n[at[:, 1], at[:, 0]] = 0

  return observation_p, observation_n


def highlight(cut, side):

  observation = Image.open("highlighted.png")
  # observation = Image.fromarray(observation)
  observation = observation.convert("RGBA")
  observation = np.array(observation)
  h, w, *_ = observation.shape
  xx, yy = np.meshgrid(np.arange(w), np.arange(h))
  index = np.column_stack([xx.flatten(), yy.flatten()])
  dot = np.dot(index.astype(float) - cut[np.newaxis, ...], cut)
  if side == +1:
    at = index[dot >= 0]
  else:
    at = index[dot < 0]
  overlay = np.zeros_like(observation)
  overlay[at[:, 1], at[:, 0]] = [255, 255, 255, 20]

  combined = Image.alpha_composite(Image.fromarray(observation), Image.fromarray(overlay))
  combined.save("highlighted.png")


class Query(BaseModel):
  ...


class Feedback(BaseModel):
  ...


class Algorithm:

  def __init__(
    self,
    observation,
    *,
    estimation_mode: str = "max",
    discount: float = 0.0,
    n_posterior_samples: int = 1_000,
    burnin: int = 10_000,
    lag: int = 10,
  ):
    self.observation = observation
    self.height, self.width, *_ = observation.shape
    self.estimation_mode = estimation_mode
    self.discount = discount
    self.n_posterior_samples = n_posterior_samples
    self.prior = vvv.random.MultivariateUniform(
      min=[0, 0],
      max=[self.width, self.height],
    )
    self._p = self.prior.prob
    self.posterior_sampler = vvv.random.MetropolisHastings(
      unnormalized_target_pdf=self.unnormalized_posterior,
      proposal=vvv.random.MultivariateUniformProposal(
        min=[0, 0],
        max=[self.width, self.height],
      ),
      burnin=burnin,
      lag=lag,
    )
    self.history = []

  def run(
    self,
    n_rounds: int = 100
  ):
    samples = self.posterior_sampler.sample(
      n_samples=self.n_posterior_samples,
      init=[
        self.width / 2,
        self.height / 2
      ],
    )
    for self.t in range(1, n_rounds + 1):
      cut = self.acquire(posterior_samples=samples)
      side = self.feedback(cut)
      self.update_posterior(cut, side)
      samples = self.posterior_sampler.sample(
        n_samples=self.n_posterior_samples,
        init=samples.mean(axis=0),
      )
      if self.estimation_mode == "mean":
        estimation = samples.mean(axis=0)
      elif self.estimation_mode == "max":
        prob = self.unnormalized_posterior(samples)
        max_prob_index = np.arange(len(prob))[prob == prob.max()]
        estimation = samples[np.random.choice(max_prob_index)]

  @abc.abstractmethod
  def acquire(self, posterior_samples, *args, **kwargs):
    ...

  @abc.abstractmethod
  def feedback(self, query, *args, **kwargs):
    ...

  @abc.abstractmethod
  def likelihood(self, x, query, feedback):
    """Calculates the likelihood of observing feedback given a query and a hypothesis.

    This method computes P(feedback | query, x), where 'x' represents a
    hypothesized "true" state (e.g., a pixel coordinate).

    Args:
      x (np.ndarray): An array of pixel coordinates.
        Its shape is `(..., 2)`, where `...` denotes any number of
        leading dimensions (`x_dims`).
      query (np.ndarray): An array of `Query` objects.
        The elements can be of any typeinheriting from a base `Query` class.
        Its shape is `(...,)`, where `...` denotes the query dimensions
        (`query_dims`).
      feedback (np.ndarray): An array of `Feedback` objects, or a single
        `Feedback` object. Its shape must be either the same as `query`'s 
        shape or an empty tuple `()`.

    Returns:
      np.ndarray: An array of likelihood values. The output shape is 
        `(x_dims..., query_dims...)`. The value at each index is calculated 
        as follows:
        - If `feedback` shape is `()`:
          `output[i..., j...] = P(feedback | query[j...], x[i...])`
        - Otherwise:
          `output[i..., j...] = P(feedback[j...] | query[j...], x[i...])`
    """
    ...

  def unnormalized_posterior(self, x):
    prob = self.prior.prob(x)
    for query, feedback in self.history:
      prob = prob * self.likelihood(x, query, feedback)
    return prob

  def update_posterior(self, query, feedback):
    self.history.append((query, feedback))
    # self.cut[self.cursor, :] = query
    # self.side[self.cursor] = feedback
    # self.cursor += 1
    highlight(query, feedback)


class EquiVolumeBisection(Algorithm):

  def acquire(self, posterior_samples):
    cuts = np.array(list(
      vvv.random.divide_rect_in_two(
        x=0, y=0, w=self.width, h=self.height,
      )
      for _ in range(5_000)
    ))
    likelihood = self.likelihood(posterior_samples, cuts, side=+1)
    likelihood = np.stack(
      [likelihood, 1 - likelihood], axis=-1
    )
    score = likelihood.mean(axis=0).min(axis=-1)
    max_score_index = np.arange(len(score))[score == score[~np.isnan(score)].max()]
    return cuts[np.random.choice(max_score_index)]

  def feedback(self, query):
    imgs = get_query(observation=self.observation, cut=query)
    Image.fromarray(imgs[0]).save("+1.png")
    Image.fromarray(imgs[1]).save("-1.png")

    image_base64 = image_to_base64("+1.png")
    response = litellm.completion(
      model="ollama/qwen2.5vl:3b",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Is there a red star in the attached image? If exists, just type 'yes', otherwise type 'no'"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
              },
            },
          ],
        }
      ],
      base_url="http://127.0.0.1:11109",
      # api_key=None,
      # mock_response=self.mock_response,
    )
    print(response.choices[0].message)

    return int(input("Which side?"))

  def likelihood(self, x, cut, side):
    x = np.asarray(x, dtype=float)
    cut = np.asarray(cut, dtype=float)
    side = np.asarray(side, dtype=float)
    assert x.shape[-1] == cut.shape[-1]
    assert side.ndim == 0 or cut.shape[:-1] == side.shape
    X = x[..., *((np.newaxis,) * len(cut.shape[:-1])), :]
    Cut = cut[*((np.newaxis,) * len(x.shape[:-1])), ..., :]
    test = side * ((X - Cut) * Cut).sum(axis=-1)
    return np.where(test >= 0, 1.0 - self.discount, self.discount)

