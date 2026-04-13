# Equation Reference

Quick lookup for all main equations: **Section A.1** = paper model (Gamus et al., arXiv:1911.05227); **Section A.2** = dynamic extension (simulation). Full derivation and context are in Sections 2–3; this appendix is for reference only.

---

## A.1 Paper model (Section 2)

### Geometry

| Quantity | Equation |
|----------|----------|
| Kinematic constraint (both feet on ground) | \(\sin(\phi_1-\theta) - \sin(\phi_2+\theta) + \beta\sin\theta = 0\) |
| Link length | \(l = L/(2+\beta)\) |
| Central link angle | \(\tan\theta = \dfrac{\sin\phi_1 - \sin\phi_2}{\cos\phi_1 + \cos\phi_2 - \beta}\) |
| Contact distance | \(d = l\bigl(\beta\cos\theta - \cos(\phi_1-\theta) - \cos(\phi_2+\theta)\bigr)\) |
| CoM offset (left contact → CoM) | \(x_c = \dfrac{l}{2(2+\beta)}\Bigl((2+\beta)\beta\cos\theta - (3+2\beta)\cos(\phi_1-\theta) - \cos(\phi_2+\theta)\Bigr)\) |

### Equilibrium

| Item | Equation |
|------|----------|
| Normal forces | \(f_{n1} = (1 - x_c/d)Mg\), \(\quad f_{n2} = (x_c/d)Mg\)  (for \(d>0\)) |
| Tangential balance | \(f_{t,1} = -f_{t,2} \equiv f_t\) |

### Slippage and slip force

| Item | Equation |
|------|----------|
| Slippage criterion | \(\Delta = x_c - d/2\)  ⇒  \(\Delta>0\): left slips; \(\Delta<0\): right slips |
| Signed slip force | \(f_t = \mu f_{n,s}\,\operatorname{sign}(\dot{d})\),  \(s=1\) if \(\Delta>0\), \(s=2\) if \(\Delta<0\) |
| Time derivative of \(d\) (paper Eq. (8)) | \(\dot{d} = l\bigl[\sin(\phi_2+\theta)(\dot{\phi}_2+\dot{\theta}) + \sin(\phi_1-\theta)(\dot{\phi}_1-\dot{\theta}) - \beta\sin\theta\,\dot{\theta}\bigr]\) |

### Kinematic update and gait

| Item | Equation |
|------|----------|
| Step displacement | \(\Delta d = d_{\mathrm{new}} - d_{\mathrm{prev}}\);  body displaced by \(\pm\Delta d\) along crawl axis |
| Reference angles | \(\phi_1^{\mathrm{ref}} = \gamma + A\sin(\omega t + \psi/2)\),  \(\phi_2^{\mathrm{ref}} = \gamma + A\sin(\omega t - \psi/2)\) |

---

## A.2 Dynamic extension (Section 3)

### Time stepping

| Item | Equation |
|------|----------|
| Linear system | \(A\,\Delta\mathbf{v} = \mathbf{f}\),  \(A = M - hD - h^2 K\) |
| Update | \(\mathbf{v}^{n+1} = \mathbf{v}^n + \Delta\mathbf{v}\),  \(\mathbf{q}^{n+1} = \mathbf{q}^n + h\,\mathbf{v}^{n+1}\) |

### Deformable body

| Item | Equation |
|------|----------|
| Spring force | \(\mathbf{F} = k_e(\ell - \ell_0)\,\hat{\mathbf{d}} + k_d\,\dot{\ell}\,\hat{\mathbf{d}}\),  \(\hat{\mathbf{d}} = (\mathbf{x}_j - \mathbf{x}_i)/\ell\) |
| Inflation (scale) | \(s = p^{1/3}\);  \(\ell_0^{\mathrm{new}} = \ell_0^{\mathrm{orig}}\,s\);  per chamber: \(s_c = p_c^{1/3}\) |

### Spine torque and ground contact

| Item | Equation |
|------|----------|
| Spine (angle \(\alpha\), not paper’s \(\theta\)) | \(\alpha = \arccos(\mathbf{d}\cdot\mathbf{d}_0)\),  \(\tau = -(k_\tau\alpha + k_d\omega)L\);  forces at endpoints from \(\tau\,\mathbf{a}\times\mathbf{d}/L\) |
| Ground normal | \(f_n = k_e c + k_d\min(\dot{c},0)\),  \(\mathbf{F}_n = -f_n\mathbf{n}\) |
| Tangential (ground) | Coulomb \(|\mathbf{F}_t| \le \mu f_{n,\mathrm{eff}}\) (opposes slip); see `07_mathematical_summary.md` §8 |
