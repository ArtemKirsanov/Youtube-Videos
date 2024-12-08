import numpy as np
from scipy.integrate import odeint


class NeuronModel:
    def __init__(self, bifurcation_type='saddle-node'):
        '''
            Simplified Hodgkin-Huxley model of a neuron with two state variables (persistent I_Na + I_K).

            Parameters are from Izhikevich, E. M. (2007). Dynamical systems in neuroscience: the geometry of excitability and bursting. MIT press.
        
        '''
        self.bifurcation_type = bifurcation_type

        assert bifurcation_type in ['saddle-node', 'SNIC', 'subcritical_Hopf', 'supercritical_Hopf']

        if bifurcation_type in ['saddle-node', 'SNIC']:
            self.potassium_threshold = 'high'
        else:
            self.potassium_threshold = 'low'

        # Membrane capacitance (μF/cm²)
        self.C_m = 1.0
        
        # Maximum conductances (mS/cm²)
        if self.bifurcation_type == 'subcritical_Hopf':
            self.g_Na = 4    
            self.g_K = 4     
            self.g_L = 1
        else:
            self.g_Na = 20    
            self.g_K = 10     
            self.g_L = 8     

        # Reversal potentials (mV)
        self.E_Na = 60.0    
        self.E_K = -90.0    

        if self.potassium_threshold == 'high':  
            self.E_L = -80
            self.V_mid_n = -25
        else:
            self.E_L = -78
            self.V_mid_n = -45
        
        # Kinetic parameters for gating variables
        if self.bifurcation_type == 'subcritical_Hopf':
            self.V_mid_m = -30
            self.k_m = 7
        else:
            self.V_mid_m = -20
            self.k_m = 15
            
        self.k_n = 5

    def m_inf(self, V):
        """Steady-state value of m (sodium activation)"""
        return 1 / (1 + np.exp(-(V - self.V_mid_m) / self.k_m))
    
    def n_inf(self, V):
       # print('V:', V)
        """Steady-state value of n (potassium activation)"""
        return 1 / (1 + np.exp(-(V - self.V_mid_n) / self.k_n))
    
    def tau_n(self, V):
        if self.bifurcation_type == 'SNIC':
            return 1
        if self.bifurcation_type == 'saddle-node':
            return 0.152
        if self.bifurcation_type == 'subcritical_Hopf':
            return 1
        if self.bifurcation_type == 'supercritical_Hopf':
            return 1
    
    def I_Na(self, V):
        """Sodium current"""
        return self.g_Na * self.m_inf(V) * (V - self.E_Na)
    
    def I_K(self, V, n):
        """Potassium current"""
        return self.g_K * n * (V - self.E_K)
    
    def I_L(self, V):
        """Leak current"""
        return self.g_L * (V - self.E_L)

    @staticmethod
    def create_step_current(t, step_time, step_duration, baseline, amplitude):
        """
        Create a step current waveform
        
        Parameters:
        -----------
        t : array-like
            Time points
        step_time : float
            Time at which step begins
        step_duration : float
            Duration of the step
        baseline : float
            Baseline current value
        amplitude : float
            Step amplitude (added to baseline)
        
        Returns:
        --------
        array-like
            Current values at each time point
        """
        I = np.ones_like(t) * baseline
        step_mask = (t >= step_time) & (t < step_time + step_duration)
        I[step_mask] = baseline + amplitude
        return I
    

    @staticmethod
    def create_ramp_current(t, ramp_start, ramp_duration, baseline, final_amplitude):
        """
        Create a linear ramp current
        
        Parameters:
        -----------
        t : array-like
            Time points
        ramp_start : float
            Time at which ramp begins
        ramp_duration : float
            Duration of the ramp
        baseline : float
            Initial current value
        final_amplitude : float
            Final current value
        
        Returns:
        --------
        array-like
            Current values at each time point
        """
        I = np.ones_like(t) * baseline
        ramp_mask = (t >= ramp_start) & (t < ramp_start + ramp_duration)
        ramp_t = t[ramp_mask] - ramp_start
        I[ramp_mask] = baseline + (final_amplitude - baseline) * (ramp_t / ramp_duration)
        I[t >= ramp_start + ramp_duration] = final_amplitude
        return I



    def find_equlibrium_points(self, I_ext, x_range, num_points=50000):
        """
        Find and analyze equilibrium points of the neuron model.
        
        Parameters:
        -----------
        I_ext : float
            External current value
        x_range : tuple or list
            (min, max) range for voltage to search for equilibria
        num_points : int, optional
            Number of points to use for nullcline calculation
            
        Returns:
        --------
        list of dicts
            Each dict contains:
            - 'point': tuple (V, n) of equilibrium point coordinates
            - 'stability': str, one of 'stable', 'unstable', or 'saddle'
            - 'eigenvalues': array of eigenvalues
            - 'jacobian': 2x2 Jacobian matrix at the point
        """
        # Find nullcline intersections
        x = np.linspace(x_range[0], x_range[1], num_points)
        v_null = self.V_nullcline(x, I_ext)
        n_null = self.n_nullcline(x)
        
        # Find where nullclines cross
        diff = v_null - n_null
        sign_changes = np.where(np.diff(np.signbit(diff)))[0]
        
        equilibria = []
        
        # Analyze each equilibrium point
        for idx in sign_changes:
            # Get precise intersection point
            x_intersect = np.interp(0, 
                                [diff[idx], diff[idx+1]], 
                                [x[idx], x[idx+1]])
            y_intersect = np.interp(x_intersect, 
                                [x[idx], x[idx+1]], 
                                [v_null[idx], v_null[idx+1]])
            
            # Calculate Jacobian numerically
            eps = 1e-8
            V, n = x_intersect, y_intersect
            
            dV1, dn1 = self.dALLdt([V+eps, n], 0, lambda t: I_ext)
            dV2, dn2 = self.dALLdt([V-eps, n], 0, lambda t: I_ext)
            dV3, dn3 = self.dALLdt([V, n+eps], 0, lambda t: I_ext)
            dV4, dn4 = self.dALLdt([V, n-eps], 0, lambda t: I_ext)
            
            J = np.array([
                [(dV1 - dV2)/(2*eps), (dV3 - dV4)/(2*eps)],
                [(dn1 - dn2)/(2*eps), (dn3 - dn4)/(2*eps)]
            ])
            
            # Get eigenvalues
            eigenvals = np.linalg.eigvals(J)
            
            # Determine stability
            if np.all(np.real(eigenvals) < 0):
                stability = 'stable'
            elif np.all(np.real(eigenvals) > 0):
                stability = 'unstable'
            else:
                stability = 'saddle'
                
            # Store results
            equilibria.append({
                'point': (V, n),
                'stability': stability,
                'eigenvalues': eigenvals,
                'jacobian': J
            })
        
        return equilibria


    def dALLdt(self, X, t, I_ext_t):
        """
        Calculate derivatives for the two state variables
        
        Parameters:
        -----------
        X : list or array
            State variables [V, n]
        t : float
            Current time
        I_ext_t : callable or array-like
            External current function or array of current values
        """
       # print('X:', X)
        V, n = X
       # print(V)
        
        # Get current value at time t
        
        if callable(I_ext_t):
            I = I_ext_t(t)
        else:
            # If I_ext_t is an array, interpolate to get current value
            idx = int(t / self.dt)  # self.dt needs to be set in simulate
            I = I_ext_t[min(idx, len(I_ext_t)-1)]
        
        # Calculate membrane potential derivative
        dVdt = (I - self.I_Na(V) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        
        # Calculate potassium gating variable derivative

        dndt = (self.n_inf(V) - n) / self.tau_n(V)
        
        return [dVdt, dndt]
    
    def dALLdt_backwards(self, X, t, I_ext_t):
        return [-x for x in self.dALLdt(X, t, I_ext_t)]

    def V_nullcline(self, V, I_ext):
        """V nullcline"""
        return (I_ext - self.I_Na(V) - self.I_L(V)) / (self.g_K * (V - self.E_K))
    
    def n_nullcline(self, V):
        """n nullcline"""
        return self.n_inf(V)


    def simulate(self, T, dt, X0, I_ext):
        """
        Basic simulation without perturbations or special pulse handling
        
        Parameters:
        -----------
        T : float
            Total simulation time
        dt : float
            Time step
        X0 : list or array
            Initial conditions [V0, n0]
        I_ext : callable or array-like
            Either a function I(t) or array of current values
        
        Returns:
        --------
        tuple
            (time points, solution array)
        """
        self.dt = dt  # Store dt for use in dALLdt
        t = np.arange(0, T, dt)
        
        # If I_ext is already an array of correct length, use it directly
        if isinstance(I_ext, (np.ndarray, list)) and len(I_ext) != len(t):
            raise ValueError("If I_ext is an array, it must have the same length as time points")
        
        # Solve ODE system
        solution = odeint(self.dALLdt, X0, t, args=(I_ext,))
        
        return t, solution

    def simulate_with_perturbations(self, T, dt, X0, I_ext, perturbations, smoothing_points=10):
        """
        Simulate with perturbations in both voltage and gating variable
        
        Parameters:
        -----------
        T : float
            Total simulation time
        dt : float
            Time step
        X0 : list or array
            Initial conditions [V0, n0]
        I_ext : callable or array-like
            Either a function I(t) or array of current values
        perturbations : list of tuples
            List of (time, deltaV, deltaN) triples specifying when and by how much to perturb
            each variable. If deltaN is None, only voltage is perturbed.
        smoothing_points : int, optional
            Number of points to add for smoothing the visualization of perturbations
            
        Returns:
        --------
        tuple
            (time points, solution array)
        """
        # Sort perturbations by time
        perturbations = sorted(perturbations, key=lambda x: x[0])
        
        # Create dense time grid including points around perturbations
        t_base = np.arange(0, T, dt)
        t_dense = list(t_base)
        
        # Add extra points around perturbations for smooth visualization
        if smoothing_points > 0:
            for pert_time, _, _ in perturbations:
                if pert_time >= T:
                    continue
                # Add points just before and after perturbation
                eps = dt / smoothing_points
                t_dense.extend([pert_time + i*eps for i in range(-smoothing_points, smoothing_points+1)])
        
        t_dense = sorted(set(t_dense))  # Remove duplicates and sort
        t_dense = np.array(t_dense)
        
        # Initialize arrays for results
        n_points = len(t_dense)
        solution = np.zeros((n_points, 2))
        
        # Initial conditions for first segment
        current_state = np.array(X0)
        
        # Current position in the output arrays
        current_idx = 0
        
        # Simulate each segment between perturbations
        for pert_time, delta_V, delta_N in perturbations:
            if pert_time >= T:
                break
                
            # Find index for this segment
            next_idx = np.searchsorted(t_dense, pert_time)
            
            # Time points for this segment
            segment_t = t_dense[current_idx:next_idx+1]
            
            if len(segment_t) > 0:
                # Simulate up to the perturbation
                segment_solution = odeint(self.dALLdt, current_state, segment_t, args=(I_ext,))
                
                # Store the results
                solution[current_idx:next_idx+1] = segment_solution
                
                # Update current state with the last point
                current_state = segment_solution[-1].copy()
            
            # Apply the perturbations
            current_state[0] += delta_V  # Voltage perturbation
            if delta_N is not None:
                current_state[1] += delta_N  # Gating variable perturbation
                
            # If smoothing is enabled, create interpolated points around perturbation
            if smoothing_points > 0:
                smooth_idx_start = next_idx - smoothing_points
                smooth_idx_end = next_idx + smoothing_points + 1
                
                if smooth_idx_start >= 0 and smooth_idx_end < len(t_dense):
                    # Create smooth transition using sigmoid function
                    smooth_t = np.linspace(-3, 3, 2*smoothing_points + 1)
                    sigmoid = 1 / (1 + np.exp(-smooth_t))
                    
                    # Interpolate between pre and post perturbation states
                    pre_state = solution[smooth_idx_start]
                    post_state = current_state
                    
                    for i, s in enumerate(sigmoid):
                        idx = smooth_idx_start + i
                        solution[idx] = pre_state + s * (post_state - pre_state)
            
            # Update index
            current_idx = next_idx + 1
        
        # Simulate the remaining time after the last perturbation
        if current_idx < n_points:
            final_t = t_dense[current_idx:]
            final_solution = odeint(self.dALLdt, current_state, final_t, args=(I_ext,))
            solution[current_idx:] = final_solution
        
        # Interpolate back to original time points if smoothing was used
        if smoothing_points > 0:
            solution_interp = np.array([
                np.interp(t_base, t_dense, solution[:, i]) 
                for i in range(2)
            ]).T
            return t_base, solution_interp
        
        return t_dense, solution


    def create_pulse_train(self, t, pulse_times, pulse_width, baseline, amplitude):
        """
        Create a train of short current pulses
        
        Parameters:
        -----------
        t : array-like
            Time points
        pulse_times : list or array
            Times at which pulses occur
        pulse_width : float
            Width of each pulse
        baseline : float
            Baseline current value
        amplitude : float
            Pulse amplitude (added to baseline)
        
        Returns:
        --------
        callable
            Function I(t) that returns current value at any time t
        """
        def I(t_eval):
            """Evaluate current at any time point"""
            # Handle both scalar and array inputs
            if isinstance(t_eval, (list, np.ndarray)):
                result = np.ones_like(t_eval) * baseline
                for t in t_eval:
                    for pulse_time in pulse_times:
                        if pulse_time <= t <= pulse_time + pulse_width:
                            idx = np.where(t_eval == t)[0]
                            result[idx] = baseline + amplitude
                return result
            else:
                # Scalar input
                for pulse_time in pulse_times:
                    if pulse_time <= t_eval <= pulse_time + pulse_width:
                        return baseline + amplitude
                return baseline
        
        # Store parameters as attributes of the function
        I.pulse_times = pulse_times
        I.pulse_width = pulse_width
        
        return I


    def simulate_with_pulses(self, T, dt, X0, I_ext):
        """
        Simulate with precise handling of brief current pulses
        
        Parameters:
        -----------
        T : float
            Total simulation time
        dt : float
            Time step
        X0 : list or array
            Initial conditions [V0, n0]
        I_ext : callable
            Current function created by create_pulse_train
        
        Returns:
        --------
        tuple
            (time points, solution array)
        """
        self.dt = dt
        t_eval = np.arange(0, T, dt)
        
        if not callable(I_ext) or not hasattr(I_ext, 'pulse_times'):
            raise ValueError("I_ext must be a function created by create_pulse_train")
        
        # Create much denser time grid around pulses
        t_dense = list(t_eval)
        eps = dt/1000  # Much smaller time offset
        n_extra = 50   # Number of extra points during pulse
        
        # Add many points during each pulse
        for t_pulse in I_ext.pulse_times:
            # Points just before and at pulse start
            t_dense.extend([t_pulse - eps, t_pulse - eps/2, t_pulse])
            
            # Multiple points during pulse
            pulse_duration = np.linspace(t_pulse, t_pulse + I_ext.pulse_width, n_extra)
            t_dense.extend(pulse_duration)
            
            # Points at and just after pulse end
            t_dense.extend([t_pulse + I_ext.pulse_width, 
                        t_pulse + I_ext.pulse_width + eps/2,
                        t_pulse + I_ext.pulse_width + eps])
        
        t_dense = sorted(set(t_dense))  # Remove duplicates and sort
        t_dense = np.array(t_dense)
        
        # Solve ODE system with dense time points and strict tolerance
        solution_dense = odeint(self.dALLdt, X0, t_dense, args=(I_ext,), 
                            rtol=1e-8, atol=1e-8, hmax=dt/10)
        
        # Interpolate back to original time points
        solution = np.array([np.interp(t_eval, t_dense, solution_dense[:, i]) 
                            for i in range(2)]).T
        
        return t_eval, solution
    

    def find_separatrix(self, I_ext, x_range=(-90, 20), num_points=10000, eps=1e-6, t_max=100, dt=0.005):
        """
        Find the separatrix by computing the stable manifold of the saddle point.
        
        Parameters:
        -----------
        I_ext : float
            External current value
        x_range : tuple
            (min, max) range for voltage to search for equilibria
        num_points : int
            Number of points to use for separatrix calculation
        eps : float
            Small perturbation size for eigenvector calculation
        t_max : float
            Maximum time for backward integration
        dt : float
            Time step for integration
            
        Returns:
        --------
        tuple
            (V_separatrix, n_separatrix) arrays containing points along the separatrix
        """
        # First find equilibrium points
        equilibria = self.find_equlibrium_points(I_ext, x_range)
        
        # Find the saddle point
        saddle_point = None
        for eq in equilibria:
            if eq['stability'] == 'saddle':
                saddle_point = eq
                break
        
        if saddle_point is None:
            raise ValueError("No saddle point found. This might not be a saddle-node bifurcation case.")
        
        # Get the stable eigenvector (corresponding to negative eigenvalue)
        eigenvals = saddle_point['eigenvalues']
        eigenvecs = np.linalg.eig(saddle_point['jacobian'])[1]
        
        stable_idx = np.argmin(np.real(eigenvals))
        stable_eigenvec = np.real(eigenvecs[:, stable_idx])
        
        # Normalize eigenvector
        stable_eigenvec = stable_eigenvec / np.linalg.norm(stable_eigenvec)
        
        # Create initial points slightly displaced from saddle point along stable eigenvector
        V_saddle, n_saddle = saddle_point['point']
        
        # We'll integrate in both directions along the eigenvector
        points_pos = np.array([V_saddle + eps * stable_eigenvec[0],
                            n_saddle + eps * stable_eigenvec[1]])
        points_neg = np.array([V_saddle - eps * stable_eigenvec[0],
                            n_saddle - eps * stable_eigenvec[1]])
        
        # Time points for backward integration
        t = np.arange(0, t_max, dt)
        
        # Integrate backwards in time (multiply derivatives by -1)
        def dALLdt_backward(X, t, I_ext_t):
            return [-x for x in self.dALLdt(X, t, I_ext_t)]
        
        # Solve for both directions
        sol_pos = odeint(dALLdt_backward, points_pos, t, args=(lambda t: I_ext,))
        sol_neg = odeint(dALLdt_backward, points_neg, t, args=(lambda t: I_ext,))
        
        # Combine solutions (reverse one of them to get continuous curve)
        V_separatrix = np.concatenate([sol_neg[::-1, 0], sol_pos[:, 0]])
        n_separatrix = np.concatenate([sol_neg[::-1, 1], sol_pos[:, 1]])
        
        mask = n_separatrix >= 0 # Only the part of the separatrix where n>0
        V_separatrix, n_separatrix = V_separatrix[mask], n_separatrix[mask]
    

        return V_separatrix, n_separatrix
    

    def find_unstable_limit_cycle(self, I_ext, dt=0.025, T_max=2000):
        '''
            Find the unstable limit cycle for the subcritical Hopf bifurcation
        '''
        self.dt = dt # Store dt for use in dALLdt
        I_ext_array = np.ones(int(T_max/dt)) * I_ext
        try:
            X0 = self.get_stable_equlibrium_location(I_ext)
            X0[0] += 0.02  # Slightly perturb the voltage
        except:
            print('No stable equilibrium found')
            raise ValueError("No stable point found. This might not be a Hopf bifurcation case below a bifurcation point.")
        
        t = np.arange(0, T_max, dt)
        solution = odeint(self.dALLdt_backwards, X0, t, args=(I_ext_array,))

        # Find peaks and return only the the last cycle
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(solution[:, 0], distance=int(0.1/dt))  # Minimum distance between peaks
        last_cycle_start = peaks[-3]
        last_cycle_end = peaks[-1]
        return solution[last_cycle_start:last_cycle_end, 0], solution[last_cycle_start:last_cycle_end, 1]


        

    def get_stable_equlibrium_location(self, I_ext):
        equlibria = self.find_equlibrium_points(I_ext, [-90, 20])
        stable_eq = [eq for eq in equlibria if eq['stability'] == 'stable'][0]
        X0 = [stable_eq['point'][0], stable_eq['point'][1]]
        return X0
    
    def get_unstable_equlibrium_location(self, I_ext):
        equlibria = self.find_equlibrium_points(I_ext, [-90, 20])
        unstable_eq = [eq for eq in equlibria if eq['stability'] == 'unstable'][0]
        X0 = [unstable_eq['point'][0], unstable_eq['point'][1]]
        return X0
    
    def get_saddle_equlibrium_location(self, I_ext):
        equlibria = self.find_equlibrium_points(I_ext, [-90, 20])
        try:
            saddle_eq = [eq for eq in equlibria if eq['stability'] == 'saddle'][0]
            X0 = [saddle_eq['point'][0], saddle_eq['point'][1]]
            return X0
        except:
            raise ValueError("No saddle point found")

    def find_limit_cycle(self, I_ext, dt=0.01, T_max=200, T_start=None):
        """
        Return one cycle of the limit cycle after discarding transients (saddle-node bifurcation)
        
        Parameters:
        -----------
        I_ext : float
            External current value
        dt : float
            Time step for simulation
        T_max : float
            Maximum simulation time
        
        Returns:
        --------
        V_cycle, n_cycle : arrays
            Voltage and gating variable traces for one cycle
        """
        # Simulate trajectory
        t = np.arange(0, T_max, dt)
        try:
            X0 = self.get_unstable_equlibrium_location(I_ext)
            X0[0] += 0.1  # Slightly perturb the voltage

            _, solution = self.simulate(T_max, dt, X0, lambda t: I_ext)
            
            if T_start is None:
                # Use only second half of simulation if T_start is not specified
                half_idx = len(t)//2
                solution = solution[half_idx:]
            else:
                # Use only data after T_start
                start_idx = np.searchsorted(t, T_start)
                solution = solution[start_idx:]
            return solution[:, 0], solution[:, 1]
        except:
            return [], [] 
        
    def find_spiking_orbit_subcritical_Hopf(self, I_ext, dt=0.01, T_max=200):
        '''
        Return one cycle of the limit cycle (subcritical Hopf bifurcation)
        '''
        # Simulate trajectory
        t = np.arange(0, T_max, dt)
        X0 = [0, 0.05]
        _, solution = self.simulate(T_max, dt, X0, lambda t: I_ext)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(solution[:, 0], distance=int(0.1/dt))  # Minimum distance between peaks
        return solution[peaks[-2]:peaks[-1], 0], solution[peaks[-2]:peaks[-1], 1]
    

    def find_invariant_circle(self, I_ext, dt=0.01, T_max=50):
        '''
        Return one cycle of the limit cycle (SNIC bifurcation)
        '''
        # Simulate trajectory
        t = np.arange(0, T_max, dt)
        X0 = self.get_saddle_equlibrium_location(I_ext)
        X0[0] += 0.1 # Slightly perturb the voltage

        _, solution = self.simulate(T_max, dt, X0, lambda t: I_ext)
        return solution[:, 0], solution[:, 1]


    def find_aligned_limit_cycle(self, I_ext, dt=0.01, T_max=200, n_cycles=3, align_phase=True):
        """
        Return phase-aligned voltage oscillations for smoother animations
        
        Parameters:
        -----------
        I_ext : float
            External current value
        dt : float
            Time step for simulation
        T_max : float
            Maximum simulation time
        n_cycles : int
            Number of cycles to return
        align_phase : bool
            Whether to align the phase of oscillations
            
        Returns:
        --------
        t, V : arrays
            Time points and voltage values for aligned oscillations
        """
        try:
            # Start near unstable equilibrium
            X0 = self.get_unstable_equlibrium_location(I_ext)
            X0[0] += 0.1  # Small perturbation
            
            # Simulate to get several cycles
            _, solution = self.simulate(T_max, dt, X0, lambda t: I_ext)
            
            # Use second half of simulation to ensure settled oscillations
            half_idx = len(solution)//2
            V = solution[half_idx:, 0]
            
            if align_phase:
                # Find peaks
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(V, distance=int(0.1/dt))  # Minimum distance between peaks
                
                if len(peaks) < 2:
                    return None, None
                    
                # Calculate period
                period = np.mean(np.diff(peaks))
                
                # Start from the first peak
                start_idx = peaks[0]
                end_idx = start_idx + int(period * n_cycles)
                
                if end_idx > len(V):
                    end_idx = len(V)
                
                V_aligned = V[start_idx:end_idx]
                t_aligned = np.arange(len(V_aligned)) * dt
                
                return t_aligned, V_aligned
            else:
                return np.arange(len(V)) * dt, V
        except:
            return None, None