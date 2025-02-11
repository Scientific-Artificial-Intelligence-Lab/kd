import numpy as np

class FiniteDifferenceOperator:
    def __init__(self, u, dx, dim):
        """
        Initializes the class with the grid values, grid spacing, and dimension.
        :param u: Input array representing the function values.
        :param dx: Grid spacing.
        :param dim: Spatial dimension.
        """
        self.u = u
        self.dx = dx
        self.dim = dim
        self.n, self.m = u.shape
        self.ux = np.zeros_like(u)

    def _finite_diff_1st(self):
        """
        Computes the first-order derivative.
        """
        self.ux[1:self.n-1, :] = (self.u[2:self.n, :] - self.u[0:self.n-2, :]) / (2 * self.dx)
        self.ux[0, :] = (-3.0 / 2 * self.u[0, :] + 2 * self.u[1, :] - 1/2 * self.u[2, :]) / self.dx
        self.ux[self.n-1, :] = (3.0 / 2 * self.u[self.n-1, :] - 2 * self.u[self.n-2, :] + 1/2 * self.u[self.n-3, :]) / self.dx

    def _finite_diff_2nd(self):
        """
        Computes the second-order derivative.
        """
        self.ux[1:self.n-1, :] = (self.u[2:self.n, :] - 2 * self.u[1:self.n-1, :] + self.u[0:self.n-2, :]) / self.dx ** 2
        self.ux[0, :] = (2 * self.u[0, :] - 5 * self.u[1, :] + 4 * self.u[2, :] - self.u[3, :]) / self.dx ** 2
        self.ux[self.n-1, :] = (2 * self.u[self.n-1, :] - 5 * self.u[self.n-2, :] + 4 * self.u[self.n-3, :] - self.u[self.n-4, :]) / self.dx ** 2

    def _finite_diff_3rd(self):
        """
        Computes the third-order derivative.
        """
        self._finite_diff_2nd()  # First compute the second-order derivative
        temp = FiniteDifferenceOperator(self.ux, self.dx, self.dim)
        temp._finite_diff_1st()  # Apply first-order difference to the second derivative
        self.ux = temp.ux

    def _finite_diff_4th(self):
        """
        Computes the fourth-order derivative.
        """
        self._finite_diff_2nd()  # First compute the second-order derivative
        temp = FiniteDifferenceOperator(self.ux, self.dx, self.dim)
        temp._finite_diff_2nd()  # Apply second-order difference to the second derivative
        self.ux = temp.ux

    def compute(self, order):
        """
        Computes the derivative of the specified order.
        :param order: Order of the derivative (1, 2, 3, 4)
        :return: The computed derivative.
        """
        if order == 1:
            self._finite_diff_1st()
        elif order == 2:
            self._finite_diff_2nd()
        elif order == 3:
            self._finite_diff_3rd()
        elif order == 4:
            self._finite_diff_4th()
        else:
            raise ValueError("Only orders 1, 2, 3, and 4 are supported.")
        return self.ux

    @staticmethod
    def diff(u, dx, order=1):
        """
        Static method for convenient direct derivative computation.
        :param u: Input array.
        :param dx: Grid spacing.
        :param order: Order of the derivative.
        :return: The computed derivative.
        """
        operator = FiniteDifferenceOperator(u, dx, dim=u.ndim)
        return operator.compute(order)
