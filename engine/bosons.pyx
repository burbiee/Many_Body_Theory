# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:12:22 2023

@author: Enea Çobo

The following Code is intended as a package for Quantum Calculations
regarding bosonic systems in second quantization form. Currently present is only
1 function for 1 specific Hamiltonian. The general Bose-Hubbard under a chemical 
potential.
"""
##################################################
##################################################
# Imports
##################################################
##################################################
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from cython cimport sizeof
from itertools import combinations_with_replacement
import numpy as np
cimport numpy as np

##################################################
##################################################
#The main structs for the bosonic state.
#It holds a pointer to an array that represents the multi state Ket
##################################################
##################################################
ctypedef struct bstate:
    int* state
    int size
    double norm_const

#free bosons from memory
cdef void free_bosons(bstate in_state) nogil:
    free(in_state.state)
    in_state.state = NULL


##################################################
##################################################
# Recursive function for Combinations with replacements formula.
# Avoids int overflow in the division the old formula suffers from
##################################################
##################################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int bosonic_basis_size(int bosons, int sites) nogil:
    if bosons < 0 or sites < 0:
        raise ValueError("Negative sites or occupations")

    cdef int result = 1
    cdef int i

    for i in range(bosons):
        result *= (sites + i)

    for i in range(1, bosons + 1):
        result //= i

    return result


## To Find the basis size of the system caller python fx
cpdef int basis_size_python(int nrbosons, int nrsites):
    return bosonic_basis_size(nrbosons, nrsites)

##################################################
##################################################
#Generate the bosons basis and order Lexicographically
#Use combinations with replecement above coded in
#the bosonic_basis_size to get the total size
##################################################
##################################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bstate* ordered_basis(int bosons, int sites):
    # Generate all possible combinations of site occupations with replacement
    combinations = list(combinations_with_replacement(range(sites), bosons))

    # Sort the combinations in lexicographic order
    combinations = sorted(combinations)
    cdef size_t length = len(combinations)

    # Create basis vectors directly from the list of states
    cdef bstate* basis_vectors = <bstate*>malloc(length * sizeof(bstate))
    
    cdef size_t i, j
    for i in range(length):
        basis_vectors[i].state = <int*>malloc(sites * sizeof(int))
        for j in range(sites):
            basis_vectors[i].state[j] = 0

        for j in combinations[i]:
            basis_vectors[i].state[j] += 1

        basis_vectors[i].size = sites
        basis_vectors[i].norm_const = 1.0
        #basis_vectors[i].norm_const = sqrt(<double>basis_vectors[i].state[site]) needed ???? check theory books

    return basis_vectors



##################################################
##################################################
#The creator/annahilator operators
#along wit the number operator
##################################################
##################################################
##
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bstate apply_creator(bstate in_state, int site) nogil:
    if site <= 0:
        raise ValueError("site number negative or 0")
    #index = site-1 due physics notation
    cdef int i
    cdef bstate cresult
    cresult.size = in_state.size
    cresult.state = <int*>malloc(in_state.size*sizeof(int))
    for i in range(in_state.size):
        cresult.state[i] = in_state.state[i]
    cresult.state[site-1] = in_state.state[site-1]+1
    cresult.norm_const = sqrt(cresult.state[site-1])*in_state.norm_const
    return cresult
##
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bstate apply_annahilator(bstate in_state, int site) nogil:
    if site <= 0:
        raise ValueError("site number negative or 0")
    #index = site-1 due physics notation
    cdef bstate aresult
    aresult.size = in_state.size
    aresult.state = <int*>malloc(in_state.size*sizeof(int))
    for i in range(in_state.size):
        aresult.state[i] = in_state.state[i]
    aresult.state[site-1] = in_state.state[site-1]-1

    if in_state.state[site-1] == 0:
        aresult.norm_const = 0.
    else:
        aresult.norm_const = sqrt(in_state.state[site-1])*in_state.norm_const
    return aresult

##
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bstate number_operator(bstate in_state, int site, int power) nogil:
    #index = site-1 due physics notation
    cdef bstate nresult
    nresult.size = in_state.size
    nresult.state = <int*>malloc(in_state.size*sizeof(int))
    for i in range(in_state.size):
        nresult.state[i] = in_state.state[i]
    nresult.state[site-1] = in_state.state[site-1]
    nresult.norm_const = <double>(nresult.state[site-1]**power)
    return nresult


##################################################
##################################################
#Functions to compare the states for Orthonormality
#And a contract function that does the inner product
#based on the orthonormality assumption between basis
#states
##################################################
##################################################
##
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint compare_states(bstate state1, bstate state2) nogil:
    if state1.size != state2.size:
        raise ValueError("states have different number of sites. direct comparison may be illogical")
    cdef int i
    for i in range(state1.size):
        if state1.state[i] != state2.state[i]:
            return 0
    return 1


##
cdef inline double contract_states(bstate state1, bstate state2) nogil:
    cdef double result
    if compare_states(state1, state2):
        result = state1.norm_const * state2.norm_const
        return result
    else:
        return 0.
    #free_bosons(state1) not needed. 
    #free_bosons(state2) not needed. will crash. free inside the fx getting called from


##################################################
##################################################
# The main calculating method for the hamiltonian and its CPython call function for future imports
# interacting_boson_gas simply takes a matrix as input and iterates through the elements
# It then changes them according to which basis states are contracted with the Hamiltonian operator
# The summation over the Hopping term and Potential term are added for each element.

# The Bose_Hubbard() creates the MeMview and starts a np array. It then runs the function above to edit it
# The basis pointer is then freed and set to NULL
##################################################
##################################################
    
##
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void interacting_boson_gas(int bosons, int sites, double t, double U, double nu, bstate* basis, double[:,::1] nparr, int h_size) nogil:
    cdef bstate tmp_state1  #troubleshoot purposes
    cdef bstate tmp_state2  #these extra tmp statements are needed
    cdef bstate tmp_state3  #because they would be performed ANYWAY
    cdef bstate tmp_state4  #if the creation/ann operators would be coded
    cdef bstate tmp_state5  #the other way %_% so better keep track here
    
    cdef int i, j, k
    for i in range(h_size):
        for j in range(h_size):
            for k in range(sites):
                tmp_state1 = (apply_creator((apply_annahilator(basis[j], (k+1)%sites+1)), (k)%sites+1))
                tmp_state2 = (apply_annahilator((apply_creator(basis[j], (k+1)%sites+1)), (k)%sites+1))
                tmp_state3 = (number_operator(basis[j], k+1, 2))
                tmp_state4 = (number_operator(basis[j], k+1, 1))
                tmp_state5 = (number_operator(basis[j], k+1, 1))
                nparr[i][j] = \
                nparr[i][j] +\
                t*contract_states(basis[i], tmp_state1) +\
                t*contract_states(basis[i], tmp_state2) +\
                U*0.5*contract_states(basis[i], tmp_state3) +\
                U*0.5*contract_states(basis[i], tmp_state4) +\
                nu*contract_states(basis[i], tmp_state5)
                free(tmp_state1.state)
                free(tmp_state2.state)
                free(tmp_state3.state)
                free(tmp_state4.state)
                free(tmp_state5.state)
    tmp_state1.state = NULL
    tmp_state2.state = NULL
    tmp_state3.state = NULL
    tmp_state4.state = NULL
    tmp_state5.state = NULL
                


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] Bose_Hubbard(int bosons, int sites, double t, double U, double nu):
    cdef int h_size = bosonic_basis_size(bosons, sites)
    cdef bstate* basis = ordered_basis(bosons, sites)
    cdef double[:,::1] hamiltonian = np.zeros((h_size, h_size), dtype=np.double)
    interacting_boson_gas(bosons, sites, t, U, nu, basis, hamiltonian, h_size)
    cdef size_t i
    for i in range(h_size):
        free_bosons(basis[i]) #remember the free_bosons accounts for setting the pointer of the bstate struct to NULL too
    free(basis)
    basis = NULL
    return hamiltonian

