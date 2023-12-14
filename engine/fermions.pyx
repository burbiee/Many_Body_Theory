# -*- coding: utf-8 -*-

"""
Created on Fri Oct 27 00:15:25 2023

@author: Enea Çobo

The following Code is intended as a package for Quantum Calculations
regarding fermionic systems in second quantization form. Currently present are
3 functions for 3 specific Hamiltonians. The general Fermi-Hubbard model, the general
Heisenberg model and the XXX model with an interaction in the z axis 
potential.

19 NOV: added the spin Fermi Hubbard
"""

##################################################
##################################################
# Imports
##################################################
##################################################
cimport cython
from libc.stdlib cimport malloc, free
from cython cimport sizeof
import numpy as np
cimport numpy as np
from math import comb


##################################################
##################################################
# The main struct for a state. with a function to print one.
# The struct holds an int with a binary representation, the physical size of the state
# and the normalization constant relevant for operators
##################################################
##################################################
ctypedef struct fstate:
    unsigned int state
    unsigned int size
    double norm_const
    
cpdef void print_fstate(fstate in_state):
    print("{State bin:",format(in_state.state, '0'+str(in_state.size)+'b'),", size: ",in_state.size,", const: ",in_state.norm_const,"}\n")
    
cdef fstate create_state(str binary, int size, double n_const):
    cdef fstate result
    
    # Check if the binary string contains only '0' and '1'
    if not all(char in '01' for char in binary):
        raise ValueError("Input string should contain only '0' and '1' characters")
    
    # Ensure size is valid (either the same as the string length or half the length)
    if size != len(binary) and size != len(binary)//2:
        raise ValueError("Invalid size. It should be equal to the length of the string or half the length")
    
    # Ensure size is the same as the length of the string for odd-sized strings
    if len(binary) % 2 == 1 and size != len(binary):
        raise ValueError("For odd-sized strings, size must be equal to the length of the string")
    
    result.state = int(binary, 2)
    result.size = size
    result.norm_const = n_const
    return result

    
##################################################
##################################################
# The functions that perform checks and operations on bits of states
# -has_occupation_nospin checks if a certain site in a string of bits is a 1 or a 0
# the site is a number that starts from 1 not 0 and counts from the left
# like in the bra-ket notation

# -flip simply flips a 1 or a 0 to its opposite in a certain position in braket notation
##################################################
##################################################
cdef inline int has_occupation_nospin(fstate in_state, int site) nogil:
    #remember the bitwise transposition below starts from the left hence the minus operation. it gives the output in improper index (starts from 1 not 0)
    return (in_state.state >> (in_state.size - site) & 1)

cdef inline int has_occupation_spin(fstate in_fstate, int site, int spin) nogil:
    if spin:
        return in_fstate.state >> (in_fstate.size - site) & 1
    else:
        return in_fstate.state >> (2*in_fstate.size - site) & 1


# Brian Kernighan's Algorithm
@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned int count_ones_nospin(fstate in_state, int in_index) nogil:
    cdef unsigned int count = 0
    cdef unsigned int mask = 1
    # Adjust the mask to consider the last site relevant digits
    mask = mask << (in_state.size - in_index)
    for i in range(in_index-1):
        mask = mask << 1
        if in_state.state & mask:
            count = count + 1
    return count


# flip a 1 to 0 or vice versa in any binary of an int
cdef fstate flip_nospin(fstate in_state, int flip_site) nogil:
    cdef fstate result
    result = in_state
    result.state = in_state.state ^ ( 1 << (in_state.size - flip_site))
    return result


##################################################
##################################################
# Same as above but these take into consideration that it is a spin basis
# therefore the initial fstate might have a number that is higher than size
# but at most 2x. Because it has to represent up and down states
# The first half the binary represents the down (0) fermions
# The second half of the binary represents the up (1) fermions
##################################################
##################################################
# Brian Kernighan's Algorithm (counting ones in a fstate)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned int count_ones_spin(fstate in_state, int in_index) nogil:
    cdef unsigned int count = 0
    cdef unsigned int mask1 = 1
    cdef unsigned int mask2 = <int>2**in_state.size
    # Adjust the mask1 to consider the last site relevant digits
    mask1 = mask1 << (in_state.size - in_index)
    mask2 = mask2 << (in_state.size - in_index)
    for i in range(in_index-1):
        mask1 = mask1 << 1
        mask2 = mask2 << 1
        if in_state.state & mask1:
            count = count + 1
        if in_state.state & mask2:
            count = count + 1
    return count


# flip_spin method returns a state with a flipped @spin at @flip_site
cdef fstate flip_spin(fstate in_state, int flip_site, int spin) nogil:
    cdef fstate result = in_state
    if spin: 
        result.state = in_state.state ^ ( 1 << (in_state.size - flip_site))
    else:
        result.state = in_state.state ^ ( 1 << (2*in_state.size - flip_site))
    return result


##################################################
##################################################
# The function to get the ordered fermion basis
# It takes a while to explain this one so take it for granted
##################################################
##################################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef fstate* get_basis(int nr_fermions, int nr_sites):
    cdef unsigned int basis_size = comb(nr_sites, nr_fermions)
    cdef fstate* basis = <fstate*>malloc(sizeof(fstate) * basis_size)
    cdef unsigned int i, state
    # Initialize state with the first combination of nr_fermions set bits
    state = (1 << nr_fermions) - 1
    cdef unsigned int x, y
    # Generate the basis states
    for i in range(basis_size):
        basis[i].state = state
        basis[i].size = nr_sites
        basis[i].norm_const = 1.0
        x = state & -state #quick cheat to avoid div by zero error. will give compiler waring with no error
        y = state + x
        state = ((state & ~y) // x) >> 1 | y
    return basis



##################################################
##################################################
# The functions for the main creation annahilation operators + others
# Very simply return a state depending on which operator acts on it
# The results are taken from Schwabl Quantum Mechanics II
##################################################
##################################################
cdef fstate apply_fcreator_nospin(fstate in_state, int site) nogil:
    cdef fstate result = in_state
    if has_occupation_nospin(in_state, site):
        result.norm_const = result.norm_const*0
    else:
        result = flip_nospin(result, site)
        result.norm_const = (-1)**(count_ones_nospin(in_state, site))
    return result



cdef fstate apply_fannahilator_nospin(fstate in_state, int site) nogil:
    cdef fstate result = in_state
    if not (has_occupation_nospin(in_state, site)):
        result.norm_const = result.norm_const*0
    else:
        result = flip_nospin(result, site)
        result.norm_const = (-1)**(count_ones_nospin(in_state, site))
    return result
    


cdef fstate number_operator_nospin(fstate in_state, int site) nogil:
    cdef fstate result = in_state
    if has_occupation_nospin(in_state, site):
        result.norm_const = result.norm_const*1.
    else:
        result.norm_const = result.norm_const*0.
    return result



##################################################
##################################################
# Same as above but considers the spin representation
##################################################
##################################################
cdef fstate apply_fcreator_spin(fstate in_fstate, int site, int spin) nogil:
    cdef fstate result = in_fstate
    if has_occupation_spin(in_fstate, site, spin):
        result.norm_const = result.norm_const*0
    else:
        result = flip_spin(in_fstate, site, spin)
        result.norm_const = result.norm_const*((-1)**(count_ones_spin(in_fstate, site)))
    return result



cdef fstate apply_fannahilator_spin(fstate in_fstate, int site, int spin) nogil:
    cdef fstate result = in_fstate
    if has_occupation_spin(in_fstate, site, spin):
        result = flip_spin(in_fstate, site, spin)
        result.norm_const = in_fstate.norm_const * ((-1)**(count_ones_spin(in_fstate, site)))
    else:
        result.norm_const = result.norm_const*0
    return result



cdef fstate number_operator_spin(fstate in_fstate, int site, int spin) nogil:
    cdef fstate result = in_fstate
    if has_occupation_spin(in_fstate, site, spin):
        result.norm_const = result.norm_const*1.
    else:
        result.norm_const = result.norm_const*0.
    return result



##################################################
##################################################
# The function that contracts Orthonormal basis states and their norm constants
##################################################
##################################################
cdef double contract_fstates(fstate in_state1, fstate in_state2) nogil:
    cdef double result = in_state1.norm_const * in_state2.norm_const
    if in_state1.state == in_state2.state:
        return result
    else:
        return 0.



##################################################
##################################################
# Here we will define methods and functions for the full Heisenberg model
# These simply include get methods for the tensort products of the pauli matrices
##################################################
##################################################
cdef double[:,::1] get_paulix_j(int j, int N):
    cdef double[:,::1] paulix = np.array([[0,1],[1,0]], dtype=np.double)
    cdef double[:,::1] result
    if (j == 1):
        result = np.kron(paulix, np.identity(<int>2**(N-1), dtype=np.double))
    elif (j!=1 and j!=N):
        result = np.kron(np.identity(<int>2**(j-1), dtype=np.int64), paulix)
        result = np.kron(result, np.identity(<int>2**(N-j), dtype=np.double))
    else:
        result = np.kron(np.identity(<int>2**(N-1), dtype=np.int64), paulix)
    return result



cdef double complex[:,::1] get_pauliy_j(int j, int N):
    cdef double complex[:,::1] pauliy = np.array([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]], dtype=np.cdouble)
    cdef double complex[:,::1] result
    if (j == 1):
        result = np.kron(pauliy, np.identity(<int>2**(N-1), dtype=np.cdouble))
    elif (j != 1 and j != N):
        result = np.kron(np.identity(<int>2**(j-1), dtype=np.int64), pauliy)
        result = np.kron(result, np.identity(<int>2**(N-j), dtype=np.cdouble))
    else:
        result = np.kron(np.identity(<int>2**(N-1), dtype=np.int64), pauliy)
    return result



cdef double[:,::1] get_pauliz_j(int j, int N):
    cdef double[:,::1] pauliz = np.array([[1,0],[0,-1]], dtype=np.double)
    cdef double[:,::1] result
    if (j == 1):
        result = np.kron(pauliz, np.identity(<int>2**(N-1), dtype=np.double))
    elif (j!=1 and j!=N):
        result = np.kron(np.identity(<int>2**(j-1), dtype=np.double), pauliz)
        result = np.kron(result, np.identity(<int>2**(N-j), dtype=np.double))
    else:
        result = np.kron(np.identity(<int>2**(N-1), dtype=np.int64), pauliz)
    return result





##################################################
##################################################
# The main calculating method for the hamiltonian and its CPython call function for future imports
# fermi_hamiltonian simply takes a matrix as input and iterates through the elements
# It then changes them according to which basis states are contracted with the Hamiltonian operator
# The summation over the Hopping term and Potential term are added for each element.

# The Fermi_Hubbard() creates the MeMview and starts a np array. It then runs the function above to edit it
# The basis pointer is then freed and set to NULL

# Below it the main call functions for the Full Heisenberg model and
# the XXX model in  a Z potential
##################################################
##################################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void Hfermi_hamiltonian_nospin(double[:,::1] arr, fstate* basis, int nr_sites, double t, double U, double chem_pot, int basis_size) nogil:
    #arr is the result the function iterates over in the call fx
    cdef int i, j
    cdef int k
    for i in range(basis_size):
        for j in range(basis_size):
            for k in range(nr_sites):
                arr[i, j] = arr[i, j] +\
                t*contract_fstates(basis[i], apply_fcreator_nospin(apply_fannahilator_nospin(basis[j], (k+2)%nr_sites), (k+1)%nr_sites)) +\
                t*contract_fstates(basis[i], apply_fcreator_nospin(apply_fannahilator_nospin(basis[j], (k+1)%nr_sites), (k+2)%nr_sites)) +\
                U*contract_fstates(basis[i], number_operator_nospin(basis[j], k+1)) +\
                chem_pot*contract_fstates(basis[i], number_operator_nospin(basis[j], k+1))

        

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void Hfermi_hamiltonian_spin(double[:,::1] arr, fstate* basis, int nr_sites, double t, double U, double chem_pot, int basis_size) nogil:
    cdef int i, j
    cdef int k
    cdef int m
    for i in range(basis_size):
        for j in range(basis_size):
            for k in range(nr_sites):
                for m in range(2):
                    arr[i, j] = arr[i, j] +\
                    t*contract_fstates(basis[i], apply_fcreator_spin(apply_fannahilator_spin(basis[j], (k+2)%nr_sites, m), (k+1)%nr_sites, m)) +\
                    t*contract_fstates(basis[i], apply_fcreator_spin(apply_fannahilator_spin(basis[j], (k+1)%nr_sites, m), (k+2)%nr_sites, m)) +\
                    U*contract_fstates(basis[i], number_operator_spin(number_operator_spin(basis[j], k+1, 0), k+1, 1)) +\
                    chem_pot*contract_fstates(basis[i], number_operator_spin(basis[j], k+1, m))
                


##################################################
##################################################
# The Python Call function for the Fermi Hubbard Model
##################################################
##################################################
cpdef double[:,::1] Fermi_Hubbard(int fermions, int sites, double t, double U, double chem_pot, str spin_status):
    cdef int assign_parameter = 0
    cdef double[:,::1] dummy_return = np.zeros((1,1))
    if spin_status == 'nospin':
        assign_parameter = 1
    elif spin_status == 'spin':
        assign_parameter = 2
    elif assign_parameter != 1 or assign_parameter != 2:
        #false return
        return dummy_return
    

    cdef fstate* basis = get_basis(fermions, assign_parameter*sites)
    cdef int f_size = comb(assign_parameter*sites, fermions)
    cdef double[:,::1] result = np.zeros((f_size, f_size), dtype=np.float64)
    
    if assign_parameter == 1:
        Hfermi_hamiltonian_nospin(result, basis, sites, t, U, chem_pot, f_size)
    elif assign_parameter == 2:
        Hfermi_hamiltonian_spin(result, basis, sites, t, U, chem_pot, f_size)
    free(basis)
    basis = NULL
    return result



##################################################
##################################################
# The Python Call functions for the full Heisenberg Model and the XXX model with Z interaction
##################################################
##################################################
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] Heisenberg(double Jx, double Jy, double Jz, double hx, double hy, double hz, int N):
    cdef double[:,::1] result = np.zeros((<int>2**N, <int>2**N), dtype=np.cdouble)
    cdef int j, nr_sites
    nr_sites = N
    for j in range(nr_sites):
        result = result +\
            Jx*np.matmul(get_paulix_j((j%N)+1, N), get_paulix_j((j+1)%N+1, N)) +\
            Jy*np.matmul(get_pauliy_j((j%N)+1, N), get_pauliy_j((j+1)%N+1, N)) +\
            Jz*np.matmul(get_pauliz_j((j%N)+1, N), get_pauliz_j((j+1)%N+1, N)) +\
            np.multiply(hx, get_paulix_j(j+1, N)) +\
            np.multiply(hy, get_pauliy_j(j+1, N)) +\
            np.multiply(hz, get_pauliz_j(j+1, N))                      
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] XXX_Z(double Jx, double hz, int N):
    cdef double[:,::1] result = np.zeros((<int>2**N, <int>2**N), dtype=np.double)
    cdef int j, nr_sites
    nr_sites = N
    for j in range(nr_sites):
        result = result +\
            Jx*np.matmul(get_paulix_j((j%N)+1, N), get_paulix_j((j+1)%N+1, N)) +\
            np.multiply(hz, get_pauliz_j(j+1, N))
    return result


