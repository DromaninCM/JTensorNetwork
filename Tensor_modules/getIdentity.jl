"""
 < Description >

 # Usage 1

 A = getIdentity(B,idB, [,idA])

 Obtain the identity tensor in the space of the idB-th leg of B. For
 example, consider a ket tensor B. Then A = getIdentity(B,2) results in:

   1      3    1       2
  -->- B ->--*-->- A ->--
       |
     2 ^
       |

 Here the numbers next to the legs mean the order of legs, and * indicates
 the location where the legs will be contracted.

 # Usage 2:
 A = getIdentity(B,idB,C,idC [,idA])

 Obtain the identity tensor in the direct product space of the Hilbert
 space of the idB-th leg of B and the space of the idC-th leg of C. For
 example, consider a ket tensor B and the identity operator C at local
 site. Then A = getIdentity(B,3,C,2) results in another ket tensor A:

   1      3    1       3
  -->- B ->--*-->- A ->--
       |           |
     2 ^         2 ^
       |           |
                   *
                 2 ^           
                   |
                   C
                   |
                 1 ^

 < Input >
 B, C : [numeric array] Tensors.
 idB, idC : [integer] Indices for B and C, respectively.

 < Option >
 idA : [interger array] If the option is given, the result A is the
       permutation of the identity tensor with the permutation index idA.
       (Default: not given, i.e., no permutation)

 < Output >
 A : [numeric array] Identity tensor. If idA option is not given, the
       1st and 2nd legs of A correspond to the idB-th leg of B and the
       idC-th leg of C, respectively. If the 'idA' option is given, the
       legs are permuted accordingly.

 Written by S.Lee (May 2, 2017); documentation edited by S.Lee (May 11,2017)
 U(1) Abelian symmetry implemented by S.Lee (Jun.23,2017)
 Julia version D. Romanin (November 6, 2020)
"""
function getIdentity(B,idB,x...)

# parsing input
if length(x)>1 # combine the spaces of two tensors
    C = x[1];
    idC = x[2];
else # consider only one space
    C = [];
    idC = [];
end

# default of options
if isempty(x) || length(x) == 2
	idA = []; # permutation of the contracted tensor (default: no permutation)
elseif isempty(C) && isa(x[1],Array)
	idA = x[1];
        x = [];
elseif  isa(x[1],Array)
	idA = x[3];
	x = [];
else
	error("ERROR: Unknown option.")
end

DB = size(B,idB);

if !isempty(C)
    DC = size(C,idC);
    A = zeros(DB*DC,DB*DC);
    for i in 1:(DB*DC)
	A[i,i] = 1.0;
    end
    A = reshape(A,(DB,DC,DB*DC));
else
    A = zeros(DB,DB);
    for i in 1:(DB)
        A[i,i] = 1.0;
    end
end

if !isempty(idA)
    if length(idA) < length(size(A))
        error("ERROR: # of elements of permutation option ''idA'' is smaller that the rank of ''A''.");
    end
    A = permutedims(A,idA);
end

A

end
