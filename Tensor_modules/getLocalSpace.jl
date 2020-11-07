"""
 < Description >

 [S,I] = getLocalSpace ('Spin',s)         % spin
 [F,Z,I] = getLocalSpace ('Fermion')      % spinless fermion
 [F,Z,S,I] = getLocalSpace ('FermionS')   % spinful (spin-1/2) fermion 

 Generates the local operators as tensors. The result operators F and S
 are rank-3, whose 1st and 3rd legs are to be contracted with bra and ket
 tensors, respectively. The 2nd legs of F and S encode the flavors of the
 operators, such as spin raising/lowering/z or particle flavor.
 Basis of the output tensors depend on the input as follows:
   * 'Spin',s: +s, +s-1, ..., -s
   * 'Fermion': |vac>, c'|vac>
   * 'FermionS': |vac>, c'_down|vac>, c'_up|vac>, c'_down c'_up|vac>
 Here c' means fermion creation operator.

 < Input >
 s : [integer or half-integer] The value of spin (e.g., 1/2, 1, 3/2, ...).

 < Output >
 S : [rank-3 tensor] Spin operators.
       S(:,1,:) : spin raising operator S_+ multiplied with 1/sqrt(2)
       S(:,2,:) : spin lowering operator S_- multiplied with 1/sqrt(2)
       S(:,3,:) : spin-z operator S_z
       Then we can construct the Heisenberg interaction (vec{S} cdot
       vec{S}) by: contract(S,3,2,conj(S),3,2) that results in
       (S^+ * S^-)/2 + (S^- * S^+)/2 + (S^z * S^z) = (S^x * S^x) + (S^y *
       S^y) + (S^z * S^z).
       There are two advantages of using S^+ and S^- rather than S^x and
       S^y: (1) more compact. For spin-1/2 case for example, S^+ and S^-
       have only one non-zero elements while S^x and S^y have two. (2) We
       can avoid complex number which can induce numerical error and cost
       larger memory; a complex number is treated as two double numbers.
 I : [rank-2 tensor] Identity operator.
 F : [rank-3 tensor] Fermion annihilation operators. For spinless fermions
       ('Fermion'), the 2nd dimension of F is singleton, and F(:,1,:) is
       the annihilation operators. For spinful fermions ('FermionS'),
       F(:,1,:) and F(:,2,:) are the annihilation operators for spin-up
       and spin-down particles, respectively.
 Z : [rank-2 tensor] Jordan-Wigner string operator for anticommutation
        sign of fermions.

 Written by S.Lee (May 4,2017)
 Updated by S.Lee (May 5,2017): Revised for the course SoSe 2019. The leg
       indicating operator flavor (e.g., spin raising, up-spin particle
       annihilating) comes to the 2nd place.
 Julia version by D. Romanin (November 6, 2020)
"""
function getLocalSpace(c,x...)

# parsing input
if !(c in ["Spin","Fermion","FermionS"])
    error("ERR: Input #1 should be either ''Spin'', ''Fermion'', or ''FermionS''.");
end

# Default values
s = 0;

if c == "Spin"
	if length(x)==0
		error("ERROR: For ''Spin'', input #2 is required.");
	end
	s = x[1];
	if (abs(2*s - round(2*s)) > 1e-14) || (s <= 0)
		error("ERROR: Input #2 for ''Spin'' should be positive (half-)integer.");
	end
	s = round(2*s)/2;
        isFermion = false;
        isSpin = true; # create S tensor
        Id = zeros(Int(2*s+1),Int(2*s+1));
	for i in (1:Int(2*s+1))
		Id[i,i] = 1.0;
	end
elseif c == "Fermion"
	isFermion = true; # create F and Z tensors
        isSpin = false;
	Id = zeros(2,2);
        for i in (1:2)
                Id[i,i] = 1.0;
        end
else
	isFermion = true;
        isSpin = true;
        s = 0.5;
        Id = zeros(4,4);
        for i in (1:4)
                Id[i,i] = 1.0;
        end
end
###

if isFermion
    if isSpin # spinful fermion
        # basis: empty, down, up, two (= c_down^+ c_up^+ |vac>)
        F = zeros(4,2,4);
        # spin-up annihilation
        F[1,1,3] =  1.0;
        F[2,1,4] = -1.0; # -1 sign due to anticommutation
        # spin-down annihilation
        F[1,2,2] =  1.0;
        F[3,2,4] =  1.0;

	Z = zeros(4,4);
	Z[1,1] = Z[4,4] =  1.0;
	Z[2,2] = Z[3,3] = -1.0;

        S = zeros(4,3,4);
        S[3,1,2] = 1/sqrt(2); # spin-raising operator (/sqrt(2))
        S[2,2,3] = 1/sqrt(2); # spin-lowering operator (/sqrt(2))
        # spin-z operator
        S[3,3,3] = 1/2;
        S[2,3,2] = -1/2;
    else # spinless fermion
        # basis: empty, occupied
        F = zeros(2,1,2);
        F[1,1,2] = 1;

	Z = zeros(2,2);
	Z[1,1] = 1.0;
	Z[2,2] = -1.0;
    end
else # spin
    # basis: (
    # spin raising operator
    rng = (s-1:-1:-s);
    flag = sqrt.((s.-rng).*(s.+rng.+1));
    Sp = zeros(Int(s*2+1),Int(s*2+1));
    for i in 1:length(flag)
	Sp[i,i+1] = flag[i]
    end

    # spin lowering operator
    rng = (s:-1:-s+1);
    flag = sqrt.((s.+rng).*(s.-rng.+1));
    Sm = zeros(Int(s*2+1),Int(s*2+1));
    for i in 1:length(flag)
        Sm[i+1,i] = flag[i]
    end

    Sz = Diagonal(s:-1:-s); # spin-z operator

    S = permutedims(cat(Sp./sqrt(2),Sm./sqrt(2),Sz;dims=3),(1,3,2));
end

# assign the tensors into varargout
if isFermion
    if isSpin # spinful fermion
        return F,Z,S,Id;
    else # spinless fermion
        return F,Z,Id;
    end
else # spin
    return S,Id;
end

end
