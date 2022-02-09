import numpy as np
import numpy.matlib as mat
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import List
from scipy.interpolate import interp1d


class ProMP:
    """A simplified implementation of ProMP.
    Original paper: A. Paraschos, C. Daniel, J. Peters, and G. Neumann, ‘Probabilistic Movement Primitives’, in Proceedings of the 26th International
    Conference on Neural Information Processing Systems - Volume 2, 2013, pp. 2616–2624.
    """

    def __init__(self, n_basis: int, n_dof: int, n_t: int, h: float = 0.07, f: float = 1.0):
        """A simplified implementation of ProMP.
        Original paper: A. Paraschos, C. Daniel, J. Peters, and G. Neumann, ‘Probabilistic Movement Primitives’, in Proceedings of the 26th International
        Conference on Neural Information Processing Systems - Volume 2, 2013, pp. 2616–2624.
        Args:
            n_basis (int): Number of basis functions.
            n_dof (int): Number of joints.
            n_t (int): Number of discrete time points.
            h (float, optional): Bandwidth of the basis functions. Defaults to 0.07.
            f (int, optional): Modulation factor. Defaults to 1.
        """
        self.n_basis = n_basis
        self.n_dof = n_dof
        self.n_t = n_t
        # Time step.
        self.dt = 1 / (n_t - 1)
        self.h = h
        self.f = f

    def all_phi(self) -> np.ndarray:
        """Build the block-diagonal matrix for all DOFs."""

        # phi has shape (T, n_basis).
        phi = self.basis_func_gauss_glb()
        # all_phi has shape (T * n_dof, n_basis * n_dof).
        all_phi = np.kron(np.eye(self.n_dof, dtype=int), phi)
        return all_phi.astype('float64')

    def basis_func_gauss_glb(self) -> np.ndarray:
        """Evaluates Gaussian basis functions in [0,1].
        This is used globally in the loss function.
        Returns:
            np.ndarray: The basis functions phi with shape (T, n_basis).
        """

        tf_ = 1/self.f
        T = int(round(tf_/self.dt+1))
        F = np.zeros((T, self.n_basis))
        for z in range(0, T):
            t = z*self.dt
            q = np.zeros((1, self.n_basis))
            for k in range(1, self.n_basis + 1):
                c = (k - 1) / (self.n_basis - 1)
                q[0, k - 1] = np.exp(-(self.f * t - c) * (self.f * t - c) / (2 * self.h))
            F[z, :self.n_basis] = q[0, :self.n_basis]

        # Normalize basis functions
        F = F / np.transpose(mat.repmat(np.sum(F, axis=1), self.n_basis, 1))
        # The output has shape (T, n_basis).
        return F.astype('float64')

    def basis_func_gauss_local(self, T: int) -> np.ndarray:
        """Evaluates Gaussian basis functions in [0,1].
        This is used for each trajectory.
        Args:
            T (int): Number of discrete time instants.
        Returns:
            np.ndarray: The basis functions phi with shape (T, n_basis).
        """

        dt = 1/(T-1)
        F = np.zeros((T, self.n_basis))
        for z in range(0, T):
            t = z*dt
            q = np.zeros((1, self.n_basis))
            for k in range(1, self.n_basis + 1):
                c = (k - 1) / (self.n_basis - 1)
                q[0, k - 1] = np.exp(-(self.f * t - c) * (self.f * t - c)/(2 * self.h))
            F[z, :self.n_basis] = q[0, :self.n_basis]

        # Normalize the basis functions.
        F = F / np.transpose(mat.repmat(np.sum(F, axis=1), self.n_basis, 1))
        # The output has shape (T, n_basis).
        return F.astype('float64')

    def weights_from_trajectory(self, trajectory: np.ndarray, vector_output: bool = True) -> np.ndarray:
        """Calculate the weights corresponding to a trajectory.
        Only the expected value is calculated.
        Args:
            trajectory (np.ndarray): Time history of each dof with shape (samples, n_dof).
            vector_output (bool, optional): If True the ouput is given in vector shape (n_dof * n_basis, ). \
                                            If False it is given in matrix shape (n_dof, n_basis). \
                                            Defaults to True.
        Returns:
            np.ndarray: The ProMP weights in a (n_dof * n_basis, ) vector or in a (n_dof, n_basis) matrix.
        """
        num_samples = trajectory.shape[0]
        num_dof= trajectory.shape[1]
        n_basis=self.basis_func_gauss_local(num_samples).shape[1]
        phi = self.basis_func_gauss_local(num_samples)  # (n_samples, n_basis)

        a=np.linalg.inv(np.matmul(phi.T, phi)+ 1e-12 * np.eye(n_basis) )
        weights =np.matmul(a,np.matmul(phi.T,trajectory)).T
        if vector_output:
            # Reshape matrix as vector.
            return weights.reshape((-1, )).astype('float64')  # (n_dof * n_basis, )
        else:
            # Keep matrix shape.
            return weights.astype('float64')  # (n_dof, n_basis)

    def trajectory_from_weights(self, weights: np.ndarray, vector_output: bool = False) -> np.ndarray:
        """Calculate the trajectory of all dofs from the given weights.
        Args:
            weights (np.ndarray): The ProMP weights with shape (n_basis * n_dof, ).
            vector_output (bool, optional): If True the ouput is given in vector shape (n_dof * n_t, ). \
                                            If False it is given in matrix shape (n_dof, n_t). \
                                            Defaults to False.
        Returns:
            np.ndarray: The trajectories of all DOFs in a (n_t, n_dof) matrix or in a (n_t * n_dof, ) vector.
        """
        trajectory = np.matmul(self.all_phi(), np.transpose(weights))
        if vector_output:
            # Keep vector shape.
            return trajectory.astype('float64')  # (n_t * n_dof, )
        else:
            # Reshape into a matrix.
            return np.reshape(trajectory, (self.n_t, -1), order='F').astype('float64') # (n_t, n_dof)

    def get_mean_from_weights(self, weights: np.ndarray):
        """
        return the mean of the weights give with (n_samples,n_basis * n_dof) shape
        """
        return np.mean(weights,axis=0,dtype='float64')
    def get_cov_from_weights(self, weights: np.ndarray):
        """
        weights: Each row of m represents a variable, and each column a single observation of all those variables.
        """
        return np.cov(weights, dtype='float64')
    def get_traj_cov(self,weights_covariance:np.ndarray):
        """
        return the covariance of a trajectory
        """
        return np.matmul(self.all_phi(), np.matmul(weights_covariance, self.all_phi().T)).astype('float64')

    def get_std_from_covariance(self,covariance:np.ndarray):
        """
        standard deviation of a trajectory
        """
        std = np.sqrt(np.diag(covariance))
        return std.astype('float64')


class ProMPTuner:

    def __init__(self, trajectories: List[np.ndarray], promp: ProMP) -> None:
        self.promp = promp
        self.trajectories = trajectories  # [(n_samples, n_dof)]

        # Llinearly interpolate the trajectories to match teh outupt of the ProMP.
        self.trajectories_interpolated = []
        for traj in self.trajectories:
            traj_interpolator = interp1d(np.linspace(0, 1, traj.shape[0]), traj, axis=0)
            self.trajectories_interpolated.append(traj_interpolator(np.linspace(0, 1, promp.n_t)))

    def tune_n_basis(self, min: int = 2, max: int = 10, step: int = 1):
        assert 2 <= min <= max, "'min' should be between 2 and 'max'"
        assert step > 0, "'step' should be > 0"

        n_traj = len(self.trajectories)
        n_basis_options = range(min, max+1, step)
        mse = np.zeros_like(n_basis_options, dtype=float)
        for i, n_basis in enumerate(n_basis_options):
            promp = ProMP(n_basis, n_dof=self.promp.n_dof, n_t=self.promp.n_t, h=self.promp.h, f=self.promp.f)
            for traj in self.trajectories_interpolated:
                traj_interpolator = interp1d(np.linspace(0, 1, traj.shape[0]), traj, axis=0)
                traj_interpolated = traj_interpolator(np.linspace(0, 1, promp.n_t))
                traj_rec = promp.trajectory_from_weights(promp.weights_from_trajectory(traj_interpolated))
                mse[i] += np.mean((traj_interpolated - traj_rec)**2)
            mse[i] /= n_traj

        print("n_basis: mse(trajectory)")
        for n_basis, mse_val in zip(n_basis_options, mse):
            print(f"{n_basis}: {mse_val:.3e}")
        plt.plot(n_basis_options, mse, 'o-')

        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    import numpy as np

    N_DOF = 3
    N_BASIS = 7
    N_T = 100

    def random_polinomial(t: np.ndarray, n_zeros: int = 1, scale: float = 1.0, y_offset: float = 0.0) -> np.ndarray:
        """Sample a random polynomial function.
        The polynomial will have n_zeros zeros uniformly sampled between min(t) and max(t).
        By default the polynomial will have co-domain [0, 1], but this can be changed with the scale and y_offset arguments.
        Args:
            t (np.ndarray): Time vector.
            n_zeros (int, optional): Number of zeros of the polynomial. Defaults to 1.
            scale (float, optional): Scale of the polynomial. Defaults to 1.0.
            y_offset (float, optional): Offset from y=0. Defaults to 0.0.
        Returns:
            np.ndarray: The polynomial sampled on t.
        """
        zeros = np.random.uniform(np.min(t), np.max(t), n_zeros)
        y = np.ones_like(t)
        for t_0 in zeros:
            y *= t - t_0
        y_min = np.min(y)
        y_max = np.max(y)
        return y_offset + (y - y_min) / (y_max - y_min) * scale

    # Generate random trajectories.
    t = np.linspace(0, 1, N_T)
    traj = np.zeros((N_T, N_DOF))
    for i in range(N_DOF):
        traj[:, i] = random_polinomial(t, n_zeros=4, scale=5)

    # Initalize the ProMP
    promp = ProMP(N_BASIS, N_DOF, N_T)
    print(promp.all_phi())
    # Compute ProMP weights.
    promp_weights = promp.weights_from_trajectory(traj)

    # Reconstruct the trajectories.
    traj_rec = promp.trajectory_from_weights(promp_weights)

    # Show a comparison between original and reconstructed trajectories.
    fig, axs = plt.subplots(N_DOF, 1)
    for i in range(N_DOF):
        axs[i].plot(t, traj[:, i], t, traj_rec[:, i])
        axs[i].legend(("Original", "Reconstructed"))
    plt.show()

    # Tune the n_basis parameter.
    promp_tuner = ProMPTuner(np.expand_dims(traj, axis=0), promp)
    promp_tuner.tune_n_basis()

    # Wsamples: evey column is a set of N=15 weights. Each set is obtained from an obstervation. There are 5 observations in total
    N_DOF = 1
    N_BASIS = 15
    N_T = 100
    promp=ProMP(N_BASIS,N_DOF,N_T)
    Wsamples = np.array([[0.0141,0.0130,0.0038,0.0029,0.0143],
                         [0.0044,0.2025,0.0178,0.0703,0.0143],
                         [0.0388,0.1042,0.0531,0.0854,0.1479],
                         [0.0025,0.0321,0.0235,0.0495,0.0086],
                         [0.0810,0.0178,0.1500,0.0310,0.0843],
                         [0.0658,0.1258,0.0488,0.1650,0.1398],
                         [0.1059,0.0821,0.0116,0.2260,0.0531],
                         [0.0032,0.0952,0.0305,0.2220,0.0025],
                         [0.2031,0.1665,0.1430,0.0842,0.0656],
                         [0.0491,0.1543,0.1232,0.1505,0.0049],
                         [0.1914,0.0525,0.0783,0.0009,0.0292],
                         [0.0584,0.1035,0.0830,0.0305,0.1452],
                         [0.0157,0.1713,0.2550,0.0695,0.0051],
                         [0.2106,0.0630,0.0942,0.0086,0.1512],
                         [0.0959,0.2093,0.1388,0.0566,0.0819]])
    weights_mean= promp.get_mean_from_weights( Wsamples.T)
    print('Weights mean shape:   ',weights_mean.shape)
    weights_cov= promp.get_cov_from_weights(Wsamples)
    print('Weights covariance shape:   ',weights_cov.shape)
    traj_mean=promp.trajectory_from_weights(weights_mean)
    print('Traj mean shape:     ',traj_mean.shape)
    traj_cov=promp.get_traj_cov(weights_cov)
    print('Traj covarinace shape:     ',traj_cov.shape)
    traj_std=promp.get_std_from_covariance(traj_cov)
    print('Traj std shape:     ',traj_std.shape)