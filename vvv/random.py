
import numpy as np


MODES = ["non-parallel", "parallel", "alter-parallel"]
current_axis = 'y'

def divide_rect_in_two(x, y, w, h, mode="y"):
  """
  주어진 직사각형을 균일하게 통과하는 임의의 직선을 샘플링합니다.
  직사각형은 좌측 상단 점 (x, y)를 기준으로 정의됩니다.

  Args:
      rect_params (tuple): (x, y, width, height) 형식의 직사각형 파라미터.
                            (x, y)는 좌측 상단 모서리의 좌표입니다.

  Returns:
      tuple: 직선을 정의하는 두 점 (p1, p2)의 튜플.
              각 점은 (x, y) 좌표 튜플입니다.
  """
  if mode == "non-parallel":
    perimeter = 2 * (w + h)

    # 둘레 상의 한 점의 좌표를 계산하는 내부 함수 (좌측 상단 기준)
    def get_point_on_perimeter(distance):
      if distance < w:
        # 1. 상단 (Top edge): 좌 -> 우
        return (x + distance, y), 1
      elif distance < w + h:
        # 2. 우측 (Right edge): 상 -> 하
        return (x + w, y + (distance - w)), 2
      elif distance < 2 * w + h:
        # 3. 하단 (Bottom edge): 우 -> 좌
        return (x + w - (distance - (w + h)), y + h), 3
      else:
        # 4. 좌측 (Left edge): 하 -> 상
        return (x, y + h - (distance - (2 * w + h))), 4

    # 둘레 위에서 서로 다른 두 개의 랜덤한 거리를 선택
    while True:
      d1, d2 = np.random.uniform(0, perimeter, 2)
      p1, side1 = get_point_on_perimeter(d1)
      p2, side2 = get_point_on_perimeter(d2)
      # 두 점 사이의 거리가 매우 작으면 다시 샘플링
      if side1 != side2 and np.linalg.norm(np.array(p1) - np.array(p2)) > 1e-6:
        p1, p2 = np.asarray(p1), np.asarray(p2)
        d = p2 - p1
        a = -p1
        p = np.dot(a, d) / np.dot(d, d) * d
        return p1 + p
  elif mode == "parallel":
    # 직사각형을 수평 또는 수직으로 나누는 경우
    if np.random.rand() < 0.5:
        # 수평선
        cut_y = y + np.random.uniform(0, h)
        p1 = (x, cut_y)
        p2 = (x + w, cut_y)
    else:
        # 수직선
        cut_x = x + np.random.uniform(0, w)
        p1 = (cut_x, y)
        p2 = (cut_x, y + h)
    p1, p2 = np.asarray(p1), np.asarray(p2)
    d = p2 - p1
    a = -p1
    p = np.dot(a, d) / np.dot(d, d) * d
    result = p1 + p
    return result
  elif mode == 'y':
    # 수직선 생성
    cut_x = x + np.random.uniform(0, w)
    p1 = (cut_x, y)
    p2 = (cut_x, y + h)
    p1, p2 = np.asarray(p1), np.asarray(p2)
    d = p2 - p1
    a = -p1
    p = np.dot(a, d) / np.dot(d, d) * d
    result = p1 + p
    return result
  elif mode == 'x':
    # 수평선 생성
    cut_y = y + np.random.uniform(0, h)
    p1 = (x, cut_y)
    p2 = (x + w, cut_y)
    p1, p2 = np.asarray(p1), np.asarray(p2)
    d = p2 - p1
    a = -p1
    p = np.dot(a, d) / np.dot(d, d) * d
    result = p1 + p
    return result
  raise


class MultivariateUniform:

  def __init__(self, min, max):
    self.min = np.asarray(min)
    self.max = np.asarray(max)
    assert self.min.shape == self.max.shape
    self.volume = np.prod(self.max - self.min)

  def prob(self, x):
    x = np.asarray(x)
    return np.where(
      np.all((x >= self.min) & (x <= self.max), axis=-1),
      1.0 / self.volume,
      0.0
    )

  def sample(self, shape=()):
    shape = (shape,) if isinstance(shape, int) else shape
    return np.random.uniform(
      self.min, self.max, size=(*shape, *self.min.shape)
    )


class MultivariateUniformProposal(MultivariateUniform):

  def prob(self, x, given):
    return super().prob(x) # we ingore `given` parameter.

  def sample(self, given, shape=()):
    return super().sample(shape) # we ingore `given` parameter.


class MetropolisHastings:

  def __init__(
    self, 
    unnormalized_target_pdf, 
    proposal, 
    burnin, 
    lag
  ):
    self.unnormalized_target_pdf = unnormalized_target_pdf
    self.proposal = proposal
    self.burnin = burnin
    self.lag = lag

  def sample(self, init, n_samples):
    samples = []
    curr = np.asarray(init)
    for _ in range(self.burnin + self.lag * n_samples - 1):
      next = self.proposal.sample(given=curr)
      metropolis_ratio = (
        self.unnormalized_target_pdf(next) / 
        self.unnormalized_target_pdf(curr)
      )
      hastings_ratio = (
        self.proposal.prob(curr, given=next) /
        self.proposal.prob(next, given=curr)
      )
      if (
        np.random.uniform(0, 1) < min(
          1, metropolis_ratio * hastings_ratio
        )
      ):
        curr = next
      samples.append(curr)
    return np.stack(samples[self.burnin::self.lag], axis=0)