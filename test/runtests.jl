using LinearAlgebra, Random
import .SkewLinearAlgebra as SLA
using Test

Random.seed!(314159) # use same pseudorandom stream for every test

@testset "README.md" begin # test from the README examples
    A = [0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]
    @test SLA.isskewhermitian(A)
    A = SLA.SkewHermitian(A)
    @test tr(A) == 0
    @test det(A) ≈ 81
    @test SLA.isskewhermitian(inv(A))
    @test inv(A) ≈ [0 1 -3 -8; -1 0 4 7; 3 -4 0 2; 8 -7 -2 0]/9
    @test A \ [1,2,3,4] ≈ [-13,13,1,-4]/3
    v = [8.306623862918073, 8.53382018538718, -1.083472677771923]
    @test hessenberg(A).H ≈ Tridiagonal(v,[0,0,0,0.],-v)
    iλ₁,iλ₂ = 11.93445871397423, 0.7541188264752862
    @test eigvals(A) ≈ [iλ₁,iλ₂,-iλ₂,-iλ₁]*im
    @test Matrix(hessenberg(A).Q) ≈ [1.0 0.0 0.0 0.0; 0.0 -0.2407717061715382 -0.9592700375676934 -0.14774972261267352; 0.0 0.8427009716003843 -0.2821382463434394 0.4585336219014009; 0.0 -0.48154341234307674 -0.014106912317171805 0.8763086996337883]
    @test eigvals(A, 0,15) ≈ [iλ₁,iλ₂]*im
    @test eigvals(A, 1:3) ≈ [iλ₁,iλ₂,-iλ₂]*im
    @test svdvals(A) ≈ [iλ₁,iλ₁,iλ₂,iλ₂]
    C = SLA.skewchol(A)
    @test transpose(C.Rm)*C.Jm*C.Rm≈A[C.Pv,C.Pv]
end

@testset "SkewLinearAlgebra.jl" begin

    for T in (Int32,Float32,Float64,ComplexF32), n in [1, 2, 10, 11]
        if T<:Integer
            A = SLA.skewhermitian(rand(convert(Array{T},-10:10), n, n) * T(2))
        else
            A = SLA.skewhermitian(randn(T, n, n))
        end
        @test SLA.isskewhermitian(A)
        @test SLA.isskewhermitian(A.data)
        B = T(2) * Matrix(A)
        @test SLA.isskewhermitian(B)
        s = rand(T)
        @test SLA.skewhermitian(s) == imag(s)
        @test parent(A) == A.data
        @test similar(A) == SLA.SkewHermitian(zeros(T, n, n))
        @test similar(A,ComplexF64) == SLA.SkewHermitian(zeros(ComplexF64, n, n))
        @test A == copy(A)::SLA.SkewHermitian
        @test copyto!(copy(4 * A), A) == A
        @test size(A) == size(A.data)
        @test size(A, 1) == size(A.data, 1)
        @test size(A, 2) == size(A.data, 2)
        @test Array(A) == A.data
        @test tr(A) == tr(A.data)
        @test tril(A) == tril(A.data)
        @test tril(A, 1) == tril(A.data, 1)
        @test triu(A) == triu(A.data)
        @test triu(A,1) == triu(A.data, 1)
        @test (-A).data == -(A.data)
        A2 = A.data * A.data
        @test A * A == A2 ≈ Hermitian(A2)
        @test A * B == A.data * B
        @test B * A == B * A.data
        if iseven(n) # for odd n, a skew-Hermitian matrix is singular
            @test inv(A)::SLA.SkewHermitian ≈ inv(A.data)
        end
        @test (A * 2).data == A.data * T(2)
        @test (2 * A).data == T(2) * A.data
        @test (A / 2).data == A.data / T(2)
        C = A + A
        @test C.data == A.data + A.data
        B = SLA.SkewHermitian(B)
        C = A - B
        @test C.data == -A.data
        B = triu(A)
        @test B ≈ triu(A.data)
        B = tril(A, n - 2)
        @test B ≈ tril(A.data, n - 2)
        k = dot(A, A)
        @test k ≈ dot(A.data, A.data)
        if n > 1
            @test A[2,1] == A.data[2,1]
            A[n, n-1] = 3
            @test A[n, n-1] === T(3)
            @test A[n-1, n] === T(-3)
        end

        x = rand(T, n)
        y = zeros(T, n)
        mul!(y, A, x, T(2), T(0))
        @test y == T(2) * A.data * x
        mul!(y, A, x)
        @test y == A * x
        k = dot(y, A, x)
        @test k ≈ adjoint(y) * A.data * x
        k = copy(y)
        mul!(y, A, x, T(2), T(3))
        @test y ≈ T(2) * A * x + T(3) * k
        B = copy(A)
        copyto!(B, A)
        @test B == A
        B = Matrix(A)
        @test B == A.data
        C = similar(B, n, n)
        mul!(C, A, B, T(2), T(0))
        @test C == T(2) * A.data * B
        mul!(C, B, A, T(2), T(0))
        @test C == T(2) * B * A.data
        B = SLA.SkewHermitian(B)
        mul!(C, B, A, T(2), T(0))
        @test C == T(2) * B.data * A.data
        B = Matrix(A)
        @test kron(A, B) ≈ kron(A.data, B)
        A.data[n,n] = T(4)
        @test SLA.isskewhermitian(A.data) == false
        A.data[n,n] = T(0)
        A.data[n,1] = T(4)
        @test SLA.isskewhermitian(A.data) == false
        
        LU = lu(A)
        @test LU.L * LU.U ≈ A.data[LU.p,:]
        if !(T<:Integer)
            LQ = lq(A)
            @test LQ.L * LQ.Q ≈ A.data
        end
        QR = qr(A)
        @test QR.Q * QR.R ≈ A.data
        if T<:Integer
            A = SLA.skewhermitian(rand(T,n,n) * 2)
        else
            A = SLA.skewhermitian(randn(T,n,n))
        end
        if eltype(A)<:Real
            F = schur(A)
            @test A.data ≈ F.vectors * F.Schur * F.vectors'
        end
        for f in (real, imag)
            @test f(A) == f(Matrix(A))
        end
    end
end

@testset "hessenberg.jl" begin
    for T in (Int32,Float32,Float64,ComplexF32), n in [1, 2, 10, 11]
        if T<:Integer
            A = SLA.skewhermitian(rand(T(-10):T(10), n, n) * T(2))
        else
            A = SLA.skewhermitian(randn(T, n, n))
        end
        B = Matrix(A)
        HA = hessenberg(A)
        HB = hessenberg(B)
        @test Matrix(HA.H) ≈ Matrix(HB.H)
        @test Matrix(HA.Q) ≈ Matrix(HB.Q)
    end

    for T in (Int32,Float64,ComplexF32)
        A = zeros(T, 4, 4)
        A[2:4,1] = ones(T,3)
        A[1,2:4] = -ones(T,3)
        A = SLA.SkewHermitian(A)
        B = Matrix(A)
        HA = hessenberg(A)
        HB = hessenberg(B)
        @test Matrix(HA.H) ≈ Matrix(HB.H)
    end
end

@testset "eigen.jl" begin
    for T in (Int32,Float32,Float64,ComplexF32), n in [1, 2, 10, 11]
        if T<:Integer
            A = SLA.skewhermitian(rand(convert(Array{T},-10:10),n,n)* T(2))
        else
            A = SLA.skewhermitian(randn(T, n, n))
        end
        B = Matrix(A)

        valA = imag(eigvals(A))
        valB = imag(eigvals(B))
        sort!(valA)
        sort!(valB)
        @test valA ≈ valB
        Eig = eigen(A)
        valA = Eig.values
        Q2 = Eig.vectors
        valB,Q = eigen(B)
        @test Q2*diagm(valA)*adjoint(Q2) ≈ A.data
        valA = imag(valA)
        valB = imag(valB)
        sort!(valA)
        sort!(valB)
        @test valA ≈ valB
        Svd = svd(A)
        @test Svd.U * Diagonal(Svd.S) * Svd.Vt ≈ A.data
        @test svdvals(A) ≈ svdvals(B)
    end

end

@testset "exp.jl" begin
    for T in (Int32,Float32,Float64,ComplexF32), n in [1, 2, 10, 11]
        if T<:Integer
            A = SLA.skewhermitian(rand(convert(Array{T},-10:10), n, n)*T(2))
        else
            A = SLA.skewhermitian(randn(T, n, n))
        end
        B = Matrix(A)
        @test exp(B) ≈ exp(A)
        @test  Matrix(cis(A)) ≈ exp(Matrix(A)*1im)
        @test cos(B) ≈ Matrix(cos(A))
        @test sin(B) ≈  Matrix(sin(A))
        sc = sincos(A)
        @test sc[1]≈ sin(B)
        @test sc[2]≈ Hermitian(cos(B))
        @test sinh(B) ≈  Matrix(sinh(A))
        @test cosh(B) ≈  Matrix(cosh(A))
    end
    for T in (Int32 ,Float32,Float64,ComplexF32), n in [2, 10, 11]
        if T<:Integer
            A = SLA.SkewHermTridiagonal(rand(convert(Array{T},-20:20), n - 1) * T(2))
        else
            if T<:Complex
                A = SLA.SkewHermTridiagonal(rand(T, n - 1), rand(real(T), n))
            else
                A = SLA.SkewHermTridiagonal(rand(T, n - 1))
            end
        end
        B = Matrix(A)
        @test exp(B) ≈ exp(A)
        @test  Matrix(cis(A)) ≈ exp(Matrix(A)*1im)
        @test cos(B) ≈ Matrix(cos(A))
        @test sin(B) ≈  Matrix(sin(A))
        sc = sincos(A)
        @test sc[1]≈ sin(B)
        @test sc[2]≈ Hermitian(cos(B))
        @test sinh(B) ≈  Matrix(sinh(A))
        @test cosh(B) ≈  Matrix(cosh(A))
    end
end

@testset "tridiag.jl" begin
    for T in (Int32,Float32,Float64,ComplexF32), n in [1, 2, 10, 11]
        if T<:Integer
            C = SLA.skewhermitian(rand(convert(Array{T},-20:20), n, n) * T(2))
        else
            C = SLA.skewhermitian(randn(T,n,n))
        end
        A = SLA.SkewHermTridiagonal(C)
        @test SLA.isskewhermitian(A) == true
        @test Tridiagonal(A) ≈ Tridiagonal(C)

        if T<:Integer
            A = SLA.SkewHermTridiagonal(rand(convert(Array{T},-20:20), n - 1) * T(2))
            C = rand(convert(Array{T},-10:10), n, n)
            D1 =  rand(convert(Array{T},-10:10), n, n)
            x = rand(convert(Array{T},-10:10), n)
            y = rand(convert(Array{T},-10:10), n)
        else
            if T<:Complex
                A = SLA.SkewHermTridiagonal(rand(T, n - 1), rand(real(T), n))
            else
                A = SLA.SkewHermTridiagonal(rand(T, n - 1))
            end
            C = randn(T, n, n)
            D1 = randn(T, n, n)
            x = randn(T, n)
            y = randn(T, n)
        end
        D2 = copy(D1)
        B = Matrix(A)
        mul!(D1, A, C, T(2), T(1))
        @test D1 ≈ D2 + T(2) * Matrix(A) * C
        mul!(D1, A, C, T(2), T(0))
        @test size(A) == (n, n)
        @test size(A,1) == n
        if A.dvim !== nothing
            @test conj(A) == SLA.SkewHermTridiagonal(conj.(A.ev),-A.dvim)
            @test copy(A) == SLA.SkewHermTridiagonal(copy(A.ev),copy(A.dvim))
        else
            @test conj(A) == SLA.SkewHermTridiagonal(conj.(A.ev))
            @test copy(A) ==SLA.SkewHermTridiagonal(copy(A.ev))
        end
        @test real(A) == SLA.SkewHermTridiagonal(real.(A.ev))
        @test transpose(A) == -A
        @test Matrix(adjoint(A)) == adjoint(Matrix(A))
        @test Array(A) == Matrix(A)
        @test D1 ≈ T(2) * Matrix(A) * C
        @test Matrix(A + A) == Matrix( 2 * A)
        @test Matrix(A)/2 == Matrix(A / 2)
        @test Matrix(A + A) == Matrix(A * 2)
        @test Matrix(A- 2 * A) == Matrix(-A)
        if n>1
            @test dot(x, A, y) ≈ dot(x, Matrix(A), y)
        end
        if T<:Complex
            z = rand(T)
            @test A * z ≈ Tridiagonal(A) * z
            @test z * A ≈ z * Tridiagonal(A)
            @test A / z ≈ Tridiagonal(A) / z
            @test z \ A ≈ z \ Tridiagonal(A)
        end
        B = Matrix(A)
        @test tr(A) ≈ tr(B)
        @test B == copy(A) == A
        yb = rand(T, 1, n)
        if !iszero(det(Tridiagonal(A)))
            @test A \ x ≈ B \ x
            @test yb / A ≈ yb / B
            #@test A / B ≈ B / A ≈ I
        end
        @test A * x ≈ B * x
        @test yb * A ≈ yb * B
        @test B * A ≈ A * B ≈ B * B
        if n>1
            @test A[1,2] == B[1,2]
        end
        @test size(A,1) == n

        EA = eigen(A)
        EB = eigen(B)
        Q = EA.vectors
        @test eigvecs(A) ≈ Q
        @test Q * diagm(EA.values) * adjoint(Q) ≈ B
        valA = imag(EA.values)
        valB = imag(EB.values)
        sort!(valA)
        sort!(valB)
        @test valA ≈ valB
        Svd = svd(A)
        @test Svd.U * Diagonal(Svd.S) * Svd.Vt ≈ B
        @test svdvals(A) ≈ svdvals(B)
        for f in (real, imag)
            @test Matrix(f(A)) == f(B)
        end
        if n > 1
            A[2,1] = 2
            @test A[2,1] === T(2) === -A[1,2]'
        end
    end

    B = SLA.SkewHermTridiagonal([3,4,5])
    @test B == [0 -3 0 0; 3 0 -4 0; 0 4 0 -5; 0 0 5 0]
    @test repr("text/plain", B) == "4×4 SkewLinearAlgebra.SkewHermTridiagonal{$Int, Vector{$Int}, Nothing}:\n ⋅  -3   ⋅   ⋅\n 3   ⋅  -4   ⋅\n ⋅   4   ⋅  -5\n ⋅   ⋅   5   ⋅"
    C = SLA.SkewHermTridiagonal(complex.([3,4,5]), [6,7,8,9])
    @test C == [6im -3 0 0; 3 7im -4 0; 0 4 8im -5; 0 0 5 9im]
    @test repr("text/plain", C) == "4×4 SkewLinearAlgebra.SkewHermTridiagonal{Complex{$Int}, Vector{Complex{$Int}}, Vector{$Int}}:\n 0+6im  -3+0im     ⋅       ⋅  \n 3+0im   0+7im  -4+0im     ⋅  \n   ⋅     4+0im   0+8im  -5+0im\n   ⋅       ⋅     5+0im   0+9im"
end

@testset "pfaffian.jl" begin
    for n in [1, 2, 3, 10, 11]
        A = SLA.skewhermitian(rand(-10:10,n,n) * 2)
        Abig = BigInt.(A.data)
        @test SLA.pfaffian(A) ≈ SLA.pfaffian(Abig)  == SLA.pfaffian(SLA.SkewHermitian(Abig))
        if VERSION ≥ v"1.7" # for exact det of BigInt matrices
            @test SLA.pfaffian(Abig)^2 == det(Abig)
        end
        @test Float64(SLA.pfaffian(Abig)^2) ≈ (iseven(n) ? det(Float64.(A)) : 0.0)
        logpf, sign = SLA.logabspfaffian(A)
        @test SLA.pfaffian(A) ≈ sign * exp(logpf)
    end
    # issue #49
    @test SLA.pfaffian(big.([0 14 7 -10 0 10 0 -11; -14 0 -10 7 13 -9 -12 -13; -7 10 0 -4 6 -17 -1 18; 10 -7 4 0 -2 -4 0 11; 0 -13 -6 2 0 -8 -18 17; -10 9 17 4 8 0 -8 12; 0 12 1 0 18 8 0 0; 11 13 -18 -11 -17 -12 0 0])) == -119000
end

@testset "cholesky.jl" begin
    for T in (Int32, Float32, Float64), n in [1, 2, 10, 11]
        if T<:Integer
            A = SLA.skewhermitian(rand(convert(Array{T},-10:10), n, n)*T(2))
        else
            A = SLA.skewhermitian(randn(T, n, n))
        end
        C = SLA.skewchol(A)
        @test transpose(C.Rm) * C.Jm *C.Rm ≈ A.data[C.Pv, C.Pv]
        B = Matrix(A)
        C = SLA.skewchol(B)
        @test transpose(C.Rm)* C.Jm *C.Rm ≈ B[C.Pv, C.Pv]
    end
end

@testset "jmatrix.jl" begin
    for T in (Int32, Float32, Float64), n in [1, 2, 10, 11], sgn in (+1,-1)
        A = rand(T, n, n)
        J = SLA.JMatrix{T,sgn}(n)
        vec = zeros(T, n - 1)
        vec[1:2:n-1] .= -sgn
        Jtest = SLA.SkewHermTridiagonal(vec)
        @test size(J) == (n, n)
        @test size(J, 1) == n
        @test J == Matrix(J) == Matrix(Jtest) == SLA.SkewHermTridiagonal(J)
        @test A*Jtest ≈ A*J
        @test Jtest*A ≈ J*A
        Jtest2 = Matrix(J)
        @test -J == -Jtest2
        @test transpose(J) == -Jtest2 == J'
        if iseven(n)
            @test inv(J) == -Jtest2
            @test J \ A ≈ Matrix(J) \ A
            @test A / J ≈ A / Matrix(J)
        end
        for k in [-4:4; n; n+1]
            @test diag(J, k) == diag(Jtest2, k)
        end
        @test iszero(tr(J))
        @test iseven(n) == det(J) ≈ det(Jtest2)
    end
    @test repr("text/plain", SLA.JMatrix(4)) == "4×4 SkewLinearAlgebra.JMatrix{Int8, 1}:\n  ⋅  1   ⋅  ⋅\n -1  ⋅   ⋅  ⋅\n  ⋅  ⋅   ⋅  1\n  ⋅  ⋅  -1  ⋅"
end

