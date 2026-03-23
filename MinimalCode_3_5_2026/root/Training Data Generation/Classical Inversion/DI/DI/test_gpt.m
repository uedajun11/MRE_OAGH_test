% Define the parameters
L = 1.0;  % Length of the domain
N = 100;  % Number of grid points
dx = L / (N - 1);  % Grid spacing

% Construct the finite difference matrix
A = diag(ones(N-1,1),1) - 2*diag(ones(N,1)) + diag(ones(N-1,1),-1);
A = A / dx^2;

% Solve for the wave number
[eigenvectors, eigenvalues] = eig(A);
eigenvalues = diag(eigenvalues);

[~, idx] = min(eigenvalues);
k = sqrt(eigenvalues(idx));  % Extracting the wave number

disp(['Wave number (k): ', num2str(k)]);
%%
% Define the parameters
Lx = 1.0;  % Length of the domain in the x-direction
Ly = 1.0;  % Length of the domain in the y-direction
Nx = 100;  % Number of grid points in the x-direction
Ny = 100;  % Number of grid points in the y-direction
dx = Lx / (Nx - 1);  % Grid spacing in the x-direction
dy = Ly / (Ny - 1);  % Grid spacing in the y-direction

% Construct the finite difference matrices
Ax = (1 / dx^2) * (spdiags([-ones(Nx, 1), 2 * ones(Nx, 1), -ones(Nx, 1)], [-1, 0, 1], Nx, Nx));
Ay = (1 / dy^2) * (spdiags([-ones(Ny, 1), 2 * ones(Ny, 1), -ones(Ny, 1)], [-1, 0, 1], Ny, Ny));
I = speye(Nx, Ny);

% Construct the 2D Laplacian matrix
A = kron(Ay, I) + kron(I, Ax);

% Solve for the wave number
[eigenvectors, eigenvalues] = eigs(A, 1, 'SM');
k = sqrt(-eigenvalues(1, 1));  % Extracting the wave number

disp(['Wave number (k): ', num2str(k)]);
