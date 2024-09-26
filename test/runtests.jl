using LinearAlgebra, Random, SparseArrays
using SkewLinearAlgebra
using Test

Random.seed!(314159) # use same pseudorandom stream for every test

isapproxskewhermitian(A) = A ≈ -A'

@testset "README.md" begin # test from the README examples
    A = [0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]
    @test isskewhermitian(A)
    A = SkewHermitian(A)
    @test tr(A) == 0
    @test det(A) ≈ 81
    @test isskewhermitian(inv(A))
    @test inv(A) ≈ [0 1 -3 -8; -1 0 4 7; 3 -4 0 2; 8 -7 -2 0]/9
    @test A \ [1,2,3,4] ≈ [-13,13,1,-4]/3
    v = [8.306623862918073, 8.53382018538718, -1.083472677771923]
    @test hessenberg(A).H ≈ Tridiagonal(v,[0,0,0,0.],-v)
    iλ₁,iλ₂ = 11.93445871397423, 0.7541188264752862
    l = imag.(eigvals(A))
    sort!(l)
    @test l ≈ sort!([iλ₁,iλ₂,-iλ₂,-iλ₁])
    @test Matrix(hessenberg(A).Q) ≈ [1.0 0.0 0.0 0.0; 0.0 -0.2407717061715382 -0.9592700375676934 -0.14774972261267352; 0.0 0.8427009716003843 -0.2821382463434394 0.4585336219014009; 0.0 -0.48154341234307674 -0.014106912317171805 0.8763086996337883]
    @test eigvals(A, 0,15) ≈ [iλ₁,iλ₂]*im
    @test eigvals(A, 1:3) ≈ [iλ₁,iλ₂,-iλ₂]*im
    @test svdvals(A) ≈ [iλ₁,iλ₁,iλ₂,iλ₂]
    C = skewchol(A)
    @test transpose(C.R)*C.J*C.R≈A[C.p,C.p]
end

@testset "SkewLinearAlgebra.jl" begin
    for T in (Int32,Float32,Float64,ComplexF32), n in [1, 2, 10, 11]
        if T<:Integer
            A = skewhermitian!(rand(convert(Array{T},-10:10), n, n) * T(2))
        else
            A = skewhermitian!(randn(T, n, n))
        end
        @test eltype(A) === T
        @test isskewhermitian(A)
        @test isskewhermitian(A.data)
        B = T(2) * Matrix(A)
        @test isskewhermitian(B)
        s = rand(T)
        @test skewhermitian(s) == imag(s)
        @test parent(A) == A.data
        @test similar(A) == SkewHermitian(zeros(T, n, n))
        @test similar(A,ComplexF64) == SkewHermitian(zeros(ComplexF64, n, n))
        @test A == copy(A)::SkewHermitian
        @test A === SkewHermitian(A) # issue #126
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
            @test inv(A)::SkewHermitian ≈ inv(A.data)
        end
        @test (A * 2).data == A.data * T(2)
        @test (2 * A).data == T(2) * A.data
        @test (A / 2).data == A.data / T(2)
        C = A + A
        @test C.data == A.data + A.data
        B = SkewHermitian(B)
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
        B = SkewHermitian(B)
        mul!(C, B, A, T(2), T(0))
        @test C == T(2) * B.data * A.data
        B = Matrix(A)
        @test kron(A, B) ≈ kron(A.data, B)
        A.data[n,n] = T(4)
        @test isskewhermitian(A.data) == false
        A.data[n,n] = T(0)
        A.data[n,1] = T(4)
        @test isskewhermitian(A.data) == false

        LU = lu(A)
        @test LU.L * LU.U ≈ A.data[LU.p,:]
        if !(T<:Integer)
            LQ = lq(A)
            @test LQ.L * LQ.Q ≈ A.data
        end
        QR = qr(A)
        @test QR.Q * QR.R ≈ A.data
        if T<:Integer
            A = skewhermitian(rand(T,n,n) * 2)
        else
            A = skewhermitian(randn(T,n,n))
        end
        if eltype(A)<:Real
            F = schur(A)
            @test A.data ≈ F.vectors * F.Schur * F.vectors'
        end
        for f in (real, imag)
            @test f(A) == f(Matrix(A))
        end
    end

    # issue #98
    @test skewhermitian([1 2; 3 4]) == [0.0 -0.5; 0.5 0.0]

    # issue #126
    let D = Diagonal([1.0+0im]), S = SkewHermitian(fill(0+1.234im, 1,1))
        @test D * S == S * D == D \ S == S / D
    end
end

@testset "hessenberg.jl" begin
    for T in (Int32,Float32,Float64,ComplexF32), n in [1, 2, 10, 11]
        if T<:Integer
            A = skewhermitian(rand(T(-10):T(10), n, n) * T(2))
        else
            A = skewhermitian(randn(T, n, n))
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
        A = SkewHermitian(A)
        B = Matrix(A)
        HA = hessenberg(A)
        HB = hessenberg(B)
        @test Matrix(HA.H) ≈ Matrix(HB.H)
    end
end

@testset "eigen.jl" begin
    for T in (Int32,Float32,Float64,ComplexF32), n in [1, 2, 10, 11]
        if T<:Integer
            A = skewhermitian(rand(convert(Array{T},-10:10),n,n)* T(2))
        else
            A = skewhermitian(randn(T, n, n))
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
            A = skewhermitian(rand(convert(Array{T},-10:10), n, n)*T(2))
        else
            A = skewhermitian(randn(T, n, n))
        end
        B = Matrix(A)
        @test exp(B) ≈ exp(A)
        @test  cis(A) ≈ Hermitian(exp(B*1im))
        @test Hermitian(cos(B)) ≈ cos(A)
        @test skewhermitian!(sin(B)) ≈  sin(A)
        sc = sincos(A)
        @test sc[1]≈ skewhermitian!(sin(B))
        @test sc[2]≈ Hermitian(cos(B))
        @test skewhermitian!(sinh(B)) ≈  sinh(A)
        @test Hermitian(cosh(B)) ≈  cosh(A)
        if T<:Complex || iseven(n)
            @test exp(log(A)) ≈ A
        end
        if issuccess(lu(cos(B), check = false)) && issuccess(lu(det(exp(2A)+I), check = false))
            if isapproxskewhermitian(tan(B)) && isapproxskewhermitian(tanh(B))
                @test tan(B) ≈ tan(A)
                @test tanh(B) ≈ tanh(A)
            end
        end
        if issuccess(lu(sin(B), check = false)) && issuccess(lu(det(exp(2A)-I), check = false))
            try
                if isapproxskewhermitian(cot(B)) && isapproxskewhermitian(coth(B))
                    @test cot(B) ≈ cot(A)
                    @test coth(B) ≈ coth(A)
                end
            catch
            end
        end
    end
    for T in (Int32 ,Float32,Float64,ComplexF32), n in [2, 10, 11]
        if T<:Integer
            A = SkewHermTridiagonal(rand(convert(Array{T},-20:20), n - 1) * T(2))
        else
            if T<:Complex
                A = SkewHermTridiagonal(rand(T, n - 1), rand(real(T), n))
            else
                A = SkewHermTridiagonal(rand(T, n - 1))
            end
        end
        B = Matrix(A)
        @test exp(B) ≈ exp(A)
        @test  cis(A) ≈ Hermitian(exp(B*1im))
        @test Hermitian(cos(B)) ≈ cos(A)
        @test skewhermitian!(sin(B)) ≈  sin(A)
        sc = sincos(A)
        @test sc[1]≈ skewhermitian!(sin(B))
        @test sc[2]≈ Hermitian(cos(B))
        @test skewhermitian!(sinh(B)) ≈  sinh(A)
        @test Hermitian(cosh(B)) ≈  cosh(A)
        if T<:Complex || iseven(n)
            @test exp(log(A)) ≈ A
        end
        if issuccess(lu(cos(B), check = false)) && issuccess(lu(det(exp(2A)+I), check = false))
            if isapproxskewhermitian(tan(B)) && isapproxskewhermitian(tanh(B))
                @test tan(B) ≈ tan(A)
                @test tanh(B) ≈ tanh(A)
            end
        end
        if issuccess(lu(sin(B), check = false)) && issuccess(lu(det(exp(2A)-I), check = false))
            try
                if isapproxskewhermitian(cot(B)) && isapproxskewhermitian(coth(B))
                    @test cot(B) ≈ cot(A)
                    @test coth(B) ≈ coth(A)
                end
            catch
            end
        end
    end
end

@testset "tridiag.jl" begin
    for T in (Int32,Float32,Float64,ComplexF32), n in [1, 2, 10, 11]
        if T<:Integer
            C = skewhermitian(rand(convert(Array{T},-20:20), n, n) * T(2))
        else
            C = skewhermitian(randn(T,n,n))
        end
        A = SkewHermTridiagonal(C)
        @test isskewhermitian(A) == true
        @test Tridiagonal(A) ≈ Tridiagonal(C)

        if T<:Integer
            A = SkewHermTridiagonal(rand(convert(Array{T},-20:20), n - 1) * T(2))
            C = rand(convert(Array{T},-10:10), n, n)
            D1 =  rand(convert(Array{T},-10:10), n, n)
            x = rand(convert(Array{T},-10:10), n)
            y = rand(convert(Array{T},-10:10), n)
        else
            if T<:Complex
                A = SkewHermTridiagonal(rand(T, n - 1), rand(real(T), n))
            else
                A = SkewHermTridiagonal(rand(T, n - 1))
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
            @test conj(A) == SkewHermTridiagonal(conj.(A.ev),-A.dvim)
            @test copy(A) == SkewHermTridiagonal(copy(A.ev),copy(A.dvim))
        else
            @test conj(A) == SkewHermTridiagonal(conj.(A.ev))
            @test copy(A) ==SkewHermTridiagonal(copy(A.ev))
        end
        @test real(A) == SkewHermTridiagonal(real.(A.ev))
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
    B = SkewHermTridiagonal([3,4,5])
    @test B == [0 -3 0 0; 3 0 -4 0; 0 4 0 -5; 0 0 5 0]
    @test repr("text/plain", B) == "4×4 SkewHermTridiagonal{$Int, Vector{$Int}, Nothing}:\n ⋅  -3   ⋅   ⋅\n 3   ⋅  -4   ⋅\n ⋅   4   ⋅  -5\n ⋅   ⋅   5   ⋅"
    C = SkewHermTridiagonal(complex.([3,4,5]), [6,7,8,9])
    @test C == [6im -3 0 0; 3 7im -4 0; 0 4 8im -5; 0 0 5 9im]
    @test repr("text/plain", C) == "4×4 SkewHermTridiagonal{Complex{$Int}, Vector{Complex{$Int}}, Vector{$Int}}:\n 0+6im  -3+0im     ⋅       ⋅  \n 3+0im   0+7im  -4+0im     ⋅  \n   ⋅     4+0im   0+8im  -5+0im\n   ⋅       ⋅     5+0im   0+9im"
end

@testset "pfaffian.jl" begin
    # real skew-hermitian matrices
    for n in [1, 2, 3, 10, 11]
        A = skewhermitian(rand(-10:10,n,n) * 2)
        Abig = BigInt.(A.data)
        @test pfaffian(A) ≈ pfaffian(Abig)  == pfaffian(SkewHermitian(Abig))
        if VERSION ≥ v"1.7" # for exact det of BigInt matrices
            @test pfaffian(Abig)^2 == det(Abig)
        end
        @test Float64(pfaffian(Abig)^2) ≈ (iseven(n) ? det(Float64.(A)) : 0.0)
        logpf, sign = logabspfaffian(A)
        @test pfaffian(A) ≈ sign * exp(logpf)

        S = SkewHermTridiagonal(A)
        logpf, sign = logabspfaffian(S)
        @test pfaffian(S) ≈ sign * exp(logpf) ≈ sign * sqrt(det(Matrix(S)))
    end

    # complex skew-symmetric matrices
    # n=11 is a bit jinxed because of https://github.com/JuliaLang/julia/issues/54287
    for n in [1, 2, 3, 10, 20]
        A = rand((-10:10) .+ 1im * (-10:10)', n, n)
        A = (A .- transpose(A)) ./ 2
        Abig = Complex{Rational{BigInt}}.(A)
        @test pfaffian(A) ≈ SkewLinearAlgebra.exactpfaffian(Abig)
        @test pfaffian(A)^2 ≈ det(A) atol=√eps(Float64) * max(1, abs(det(A)))
        if VERSION ≥ v"1.7" # for exact det of BigInt matrices
            @test SkewLinearAlgebra.exactpfaffian(Abig)^2 == det(Abig)
        end

        logpf, sign = logabspfaffian(A)
        @test pfaffian(A) ≈ sign * exp(logpf)
    end

    # issue #49
    @test pfaffian(big.([0 14 7 -10 0 10 0 -11; -14 0 -10 7 13 -9 -12 -13; -7 10 0 -4 6 -17 -1 18; 10 -7 4 0 -2 -4 0 11; 0 -13 -6 2 0 -8 -18 17; -10 9 17 4 8 0 -8 12; 0 12 1 0 18 8 0 0; 11 13 -18 -11 -17 -12 0 0])) == -119000

    # test a few 4x4 Pfaffians against the analytical formula
    for i = 1:10
        a,b,c,d,e,f = rand(-10:10,6)
        A = [0 a b c; -a 0 d e; -b -d 0 f; -c -e -f 0]
        @test SkewLinearAlgebra.exactpfaffian(A) == c*d - b*e + a*f ≈ pfaffian(A)
    end

    # issue #121
    @test pfaffian([0 1 0 0; -1 0 0 0; 0 0 0 1; 0 0 -1 0]) == 1
end

@testset "cholesky.jl" begin
    for T in (Int32, Float32, Float64), n in [1, 2, 10, 11]
        if T<:Integer
            A = skewhermitian(rand(convert(Array{T},-10:10), n, n)*T(2))
        else
            A = skewhermitian(randn(T, n, n))
        end
        C = skewchol(A)
        @test transpose(C.R) * C.J *C.R ≈ A.data[C.p, C.p]
        B = Matrix(A)
        C = skewchol(B)
        @test transpose(C.R)* C.J *C.R ≈ B[C.p, C.p]
    end
end

@testset "jmatrix.jl" begin
    for T in (Int32, Float32, Float64), n in [1, 2, 10, 11], sgn in (+1,-1)
        A = rand(T, n, n)
        J = JMatrix{T,sgn}(n)
        vec = zeros(T, n - 1)
        vec[1:2:n-1] .= -sgn
        Jtest = SkewHermTridiagonal(vec)
        @test size(J) == (n, n)
        @test size(J, 1) == n
        @test J == Matrix(J) == Matrix(Jtest) == SkewHermTridiagonal(J)
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
    @test repr("text/plain", JMatrix(4)) == "4×4 JMatrix{Int8, 1}:\n  ⋅  1   ⋅  ⋅\n -1  ⋅   ⋅  ⋅\n  ⋅  ⋅   ⋅  1\n  ⋅  ⋅  -1  ⋅"
end

@testset "issue#116" begin

        for ev in ([0,0,0], [1,2,3,2,1], [0,1,0], [1,0,0], [1,0,1], [1,1,0], [0,1,1], [0,0,1], [1,1,1],[1,1,0,1,1],[0,0,0,0,1,1,1],[0,1,0,1,0,1,1,1,0,0,0,1])
            A = SkewHermTridiagonal(float.(ev))
            a = sort(eigvals(A), by = imag)
            b = sort(eigvals(im * Matrix(A)) / im, by = imag)
            @test a ≈ b
            E = eigen(A)
            @test E.vectors*Diagonal(E.values)*E.vectors'≈A
        end

        for ev in ([1,1,0,1],[0,1,0,1,1,1,1,1])
            A = SkewHermTridiagonal(float.(ev))
            a = sort(eigvals(A), by = imag)
            b = sort(eigvals(im * Matrix(A)) / im, by = imag)
            @test a ≈ b
            E = eigen(A)
            @test E.vectors*Diagonal(E.values)*E.vectors'≈A
        end

end

@testset "issue#118 and issue#130" begin
    #issue #130
    sp = sparse([2, 7, 1, 3, 6, 8, 2, 4, 7, 9, 3, 5, 8, 10, 4, 9, 2, 7, 1, 3, 6, 8, 2, 4, 7, 9, 3, 5, 8, 10, 4, 9], [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10], [-0.8414709848, 1.5403023059, 0.8414709848, -0.8414709848, 0.4596976941, 1.5403023059, 0.8414709848, -0.8414709848, 0.4596976941, 1.5403023059, 0.8414709848, -0.8414709848, 0.4596976941, 1.5403023059, 0.8414709848, 0.4596976941, -0.4596976941, 0.8414709848, -1.5403023059, -0.4596976941, -0.8414709848, 0.8414709848, -1.5403023059, -0.4596976941, -0.8414709848, 0.8414709848, -1.5403023059, -0.4596976941, -0.8414709848, 0.8414709848, -1.5403023059, -0.8414709848], 10, 10)
    A = SkewHermitian(Matrix(sp))
    E = eigen(A)
    @test E.vectors*Diagonal(E.values)*E.vectors' ≈ A
    sp = sparse([26, 50, 51, 52, 27, 51, 52, 53, 28, 52, 53, 54, 29, 53, 54, 55, 30, 54, 55, 56, 31, 55, 56, 32, 33, 57, 58, 32, 33, 34, 57, 58, 59, 33, 34, 35, 58, 59, 60, 34, 35, 36, 59, 60, 61, 35, 36, 37, 60, 61, 62, 36, 37, 38, 61, 62, 63, 37, 38, 39, 62, 63, 64, 38, 39, 40, 63, 64, 65, 39, 40, 41, 64, 65, 66, 40, 41, 42, 65, 66, 41, 42, 43, 66, 42, 43, 44, 43, 44, 45, 44, 45, 46, 45, 46, 47, 46, 47, 48, 47, 48, 49, 48, 49, 50, 49, 50, 51, 1, 50, 51, 52, 2, 51, 52, 53, 3, 52, 53, 54, 4, 53, 54, 55, 5, 54, 55, 56, 6, 55, 56, 7, 8, 7, 8, 9, 58, 8, 9, 10, 59, 9, 10, 11, 60, 10, 11, 12, 61, 11, 12, 13, 62, 12, 13, 14, 63, 13, 14, 15, 64, 14, 15, 16, 65, 15, 16, 17, 66, 16, 17, 18, 17, 18, 19, 18, 19, 20, 19, 20, 21, 20, 21, 22, 21, 22, 23, 22, 23, 24, 23, 24, 25, 1, 24, 25, 26, 1, 2, 25, 26, 27, 1, 2, 3, 26, 27, 28, 2, 3, 4, 27, 28, 29, 3, 4, 5, 28, 29, 30, 4, 5, 6, 29, 30, 31, 5, 6, 30, 31, 7, 8, 7, 8, 9, 33, 8, 9, 10, 34, 9, 10, 11, 35, 10, 11, 12, 36, 11, 12, 13, 37, 12, 13, 14, 38, 13, 14, 15, 39, 14, 15, 16, 40, 15, 16, 17, 41], [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 58, 58, 58, 58, 59, 59, 59, 59, 60, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63, 63, 64, 64, 64, 64, 65, 65, 65, 65, 66, 66, 66, 66], [-1.176e-13, -0.0980580675690928, 0.987744983947536, -0.0980580675690911, -4.9e-14, -0.0980580675690911, 0.987744983947536, -0.0980580675690927, 1.96e-14, -0.0980580675690927, 0.987744983947536, -0.098058067569091, -1.96e-13, -0.098058067569091, 0.987744983947536, -0.0980580675690928, -1.274e-13, -0.0980580675690928, 0.987744983947536, -0.0980580675690928, -5.88e-14, -0.0980580675690928, 0.987744983947536, -10.0, -0.4902903378454601, -100.97143868952186, -0.098058067569092, 0.4902903378454601, -10.0, -0.4902903378454601, -0.098058067569092, -100.97143868952186, -0.098058067569092, 0.4902903378454601, -10.0, -0.4902903378454601, -0.098058067569092, -100.97143868952186, -0.0980580675690921, 0.4902903378454601, -10.0, -0.4902903378454601, -0.0980580675690921, -100.97143868952186, -0.0980580675690919, 0.4902903378454601, -10.0, -0.4902903378454601, -0.0980580675690919, -100.97143868952186, -0.0980580675690919, 0.4902903378454601, -10.0, -0.49029033784546, -0.0980580675690919, -100.97143868952186, -0.0980580675690923, 0.49029033784546, -10.0, -0.4902903378454601, -0.0980580675690923, -100.97143868952186, -0.0980580675690919, 0.4902903378454601, -10.0, -0.4902903378454601, -0.0980580675690919, -100.97143868952186, -0.0980580675690919, 0.4902903378454601, -10.0, -0.4902903378454601, -0.0980580675690919, -100.97143868952186, -0.0980580675690919, 0.4902903378454601, -10.0, -0.4902903378454601, -0.0980580675690919, -100.97143868952186, 0.4902903378454601, -10.0, -0.4902903378454601, -0.0980580675690919, 0.4902903378454601, -10.0, -0.4902903378454599, 0.4902903378454599, -10.0, -0.4902903378454603, 0.4902903378454603, -10.0, -0.4902903378454599, 0.4902903378454599, -10.0, -0.4902903378454602, 0.4902903378454603, -10.0, -0.4902903378454599, 0.4902903378454599, -10.0, -0.4902903378454599, 0.4902903378454599, -10.0, -0.4902903378454603, 0.4902903378454603, -10.0, -0.4902903378454599, 1.176e-13, 0.4902903378454599, -10.0, -0.4902903378454602, 4.9e-14, 0.4902903378454602, -10.0, -0.4902903378454599, -1.96e-14, 0.4902903378454599, -10.0, -0.4902903378454603, 1.96e-13, 0.4902903378454603, -10.0, -0.4902903378454599, 1.274e-13, 0.4902903378454599, -10.0, -0.4902903378454599, 5.88e-14, 0.4902903378454599, -10.0, 10.0, -0.4902903378454601, 0.4902903378454601, 10.0, -0.4902903378454601, 2.4e-15, 0.4902903378454601, 10.0, -0.4902903378454601, 4.9e-15, 0.4902903378454601, 10.0, -0.4902903378454601, 7.3e-15, 0.4902903378454601, 10.0, -0.4902903378454601, 9.8e-15, 0.4902903378454601, 10.0, -0.49029033784546, 1.22e-14, 0.49029033784546, 10.0, -0.4902903378454601, 1.47e-14, 0.4902903378454601, 10.0, -0.4902903378454601, 1.71e-14, 0.4902903378454601, 10.0, -0.4902903378454601, 1.96e-14, 0.4902903378454601, 10.0, -0.4902903378454601, 2.2e-14, 0.4902903378454601, 10.0, -0.4902903378454601, 0.4902903378454601, 10.0, -0.4902903378454599, 0.4902903378454599, 10.0, -0.4902903378454603, 0.4902903378454603, 10.0, -0.4902903378454599, 0.4902903378454599, 10.0, -0.4902903378454603, 0.4902903378454602, 10.0, -0.4902903378454599, 0.4902903378454599, 10.0, -0.4902903378454599, 0.4902903378454599, 10.0, -0.4902903378454603, 0.0980580675690928, 0.4902903378454603, 10.0, -0.4902903378454599, -0.987744983947536, 0.0980580675690911, 0.4902903378454599, 10.0, -0.4902903378454602, 0.0980580675690911, -0.987744983947536, 0.0980580675690927, 0.4902903378454602, 10.0, -0.4902903378454599, 0.0980580675690927, -0.987744983947536, 0.098058067569091, 0.4902903378454599, 10.0, -0.4902903378454603, 0.098058067569091, -0.987744983947536, 0.0980580675690928, 0.4902903378454603, 10.0, -0.4902903378454599, 0.0980580675690928, -0.987744983947536, 0.0980580675690928, 0.4902903378454599, 10.0, -0.4902903378454599, 0.0980580675690928, -0.987744983947536, 0.4902903378454599, 10.0, 100.97143868952186, 0.098058067569092, 0.098058067569092, 100.97143868952186, 0.098058067569092, -2.4e-15, 0.098058067569092, 100.97143868952186, 0.0980580675690921, -4.9e-15, 0.0980580675690921, 100.97143868952186, 0.0980580675690919, -7.3e-15, 0.0980580675690919, 100.97143868952186, 0.0980580675690919, -9.8e-15, 0.0980580675690919, 100.97143868952186, 0.0980580675690923, -1.22e-14, 0.0980580675690923, 100.97143868952186, 0.0980580675690919, -1.47e-14, 0.0980580675690919, 100.97143868952186, 0.0980580675690919, -1.71e-14, 0.0980580675690919, 100.97143868952186, 0.0980580675690919, -1.96e-14, 0.0980580675690919, 100.97143868952186, 0.0980580675690919, -2.2e-14], 66, 66)
    A = SkewHermitian(Matrix(sp))
    E = eigen(A)
    @test E.vectors*Diagonal(E.values)*E.vectors' ≈ A
    #issue #118
    for v ∈ ([1.0, 0.001, 1.0, 0.0001, 1.0], [2.0, 1e-11, 2.0, 1e-11, 2.0])
        A = SkewHermTridiagonal(v)
        E = eigen(A)
        @test E.vectors*Diagonal(E.values)*E.vectors' ≈ A
        B = SkewHermitian(Matrix(A))
        E = eigen(B)
        @test E.vectors*Diagonal(E.values)*E.vectors' ≈ B
    end
end

include("chainrulestests.jl")