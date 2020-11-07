"""
 < Description >

 Cleft = updateLeft(Cleft,rankC,B,X,rankX,A)

 Contract the operator Cleft that act on the Hilbert space of the left
 part of the MPS (i.e., left of a given site) with the tensors B, X, and
 A, acting on the given site.

 < Input >
 Cleft : [tensor] Rank-2 or 3 tensor from the left part of the system. If
       given as empty [], then Cleft is considered as the identity.
 rankC : [integer] Rank of Cleft.
 B, A : [tensors] Ket tensors, whose legs are ordered as left - bottom
       (local physical) - right. In the contraction, the Hermitian
       conjugate (i.e., bra form) of B is used, while A is contracted as
       it is. This convention of inputting B as a ket tensor reduces extra
       computational cost of taking the Hermitian conjugate of B.
 X : [tensor] Local operator with rank 2 or 3. If given as empty [], then
       X is considered as the identity.
 rankX : [integer] Rank of X.

 < Output >
 Cleft : [tensor] Contracted tensor. The tensor network diagrams
       describing the contraction are as follows.
       * When Cleft is rank-3 and X is rank-2:
                    1     3
          /--------->- A ->--            /---->-- 3
          |            | 2               |
        3 ^            ^                 |
          |    2       | 2               |      
        Cleft---       X         =>    Cleft ---- 2
          |            | 1               |
        1 ^            ^                 |
          |            | 2               |
          /---------<- B'-<--            /----<-- 1
                    3     1
       * When Cleft is rank-2 and X is rank-3:
                    1     3
          /--------->- A ->--            /---->-- 3
          |            | 2               |
        2 ^            ^                 |
          |          3 |   2             |      
        Cleft          X ----    =>    Cleft ---- 2
          |          1 |                 |
        1 ^            ^                 |
          |            | 2               |
          /---------<- B'-<--            /----<-- 1
                    3     1
       * When both Cleft and X are rank-3:
                    1     3
          /--------->- A ->--            /---->-- 2
          |            | 2               |
        3 ^            ^                 |
          |   2     2  | 3               |      
        Cleft--------- X         =>    Cleft
          |            | 1               |
        1 ^            ^                 |
          |            | 2               |
          /---------<- B'-<--            /----<-- 1
                    3     1
       * When Cleft is rank3 and X is rank-4:
                    1     3
          /--------->- A ->--            /---->-- 3
          |            | 2               |
        3 ^            ^                 |
          |   2    1   | 4               |      
        Cleft--------- X ---- 3   =>   Cleft ---- 2
          |            | 2               |
        1 ^            ^                 |
          |            | 2               |
          /---------<- B'-<--            /----<-- 1
                    3     1
       Here B' denotes the Hermitian conjugate (i.e., complex conjugate
       and permute legs by [3 2 1]) of B.

Written by H.Tu (May 3,2017); edited by S.Lee (May 19,2017)
Rewritten by S.Lee (May 5,2019)
Updated by S.Lee (May 27,2019): Case of rank-3 Cleft and rank-4 X is added.
Julia version by D. Romanin (November 7,2020)
"""
function updateLeft(Cleft,rankC,B,X,rankX,A)

# error checking
if !isempty(Cleft) && !any(i->(i in [2,3]), rankC)
    error("ERROR: Rank of Cleft or Cright should be 2 or 3.");
end
if !isempty(X) && !any(i->(i in (2:4)),rankX)
    error("ERROR: Rank of X should be 2, 3, or 4.");
end

B = conj(B); # take complex conjugate to B, without permuting legs

if !isempty(X)
    T = contract(X,rankX,[rankX],A,3,[2]);

    if !isempty(Cleft)
        if (rankC > 2) && (rankX > 2)
            if rankX == 4
                # contract the 2nd leg of Cleft and the 1st leg of X
                T = contract(Cleft,rankC,[2 rankC],T,rankX+1,[1 rankX]);
            else
                # contract the operator-flavor legs of Cleft and X
                T = contract(Cleft,rankC,[2 rankC],T,rankX+1,[2 rankX]);
            end
            Cleft = contract(B,3,[1 2],T,rankC+rankX-3,[1 2]);
        else
            T = contract(Cleft,rankC,[rankC],T,rankX+1,[rankX]);
            Cleft = contract(B,3,[1 2],T,rankC+rankX-1,[1 rankC]);
        end
    else
        Cleft = contract(B,3,[1 2],T,rankX+1,[rankX 1]);
    end
elseif !isempty(Cleft)
    T = contract(Cleft,rankC,[rankC],A,3,[1]);
    Cleft = contract(B,3,[1 2],T,rankC+1,[1 rankC]);
else
    Cleft = contract(B,3,[1 2],A,3,[1 2]);
end

end
