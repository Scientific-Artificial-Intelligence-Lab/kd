dlga_cases = [
    "u_t=-0.9901*u*ux+-0.0025*uxxx",
    "u_t=0.1231*ux+-0.6359*u*ux+-0.0026*ux*uxx+-0.0*ux*uxxx*uxxx+-0.0013*uxxx+0.0003*u*u*uxxx",
    "u_t=-0.0*uxxx*uxxx+-0.0011*ux*uxx+-0.8526*u*ux+0.1806*ux+-0.0015*uxxx",
    "u_t=0.1498*ux+-0.6168*u*ux+-0.0013*uxxx+-0.0*u*ux*uxx*uxx+-0.0012*ux*uxx+0.0002*u*uxxx",
    "u_t=-0.0022*uxxx+-0.8956*u*ux"
]


dprlpinn_cases = [
    "-0.9048 * mul_t(u1,diff_t(u1,x1)) + 0.0898 * diff2_t(u1,x1)",
    "0.0029 * diff2_t(mul_t(u1,n2_t(x1)),x1) + -0.1084 * n2_t(u1) + -0.3331 * diff_t(u1,x1)",
    "-0.0454 * u1 + -0.3350 * diff_t(u1,x1)",
    "-0.3424 * diff_t(u1,x1)",
    "-0.0017 * diff_t(div_t(mul_t(x1,u1),u1),x1) + 0.0357 * diff2_t(u1,x1) + -0.3414 * diff_t(u1,x1)",
    "-0.3426 * diff_t(u1,x1) + -0.0496 * u1",
    "-0.0496 * u1 + -0.3426 * diff_t(u1,x1)",
    "-0.3644 * diff_t(u1,x1) + -0.0528 * u1",
    "-0.0509 * u1 + -0.8952 * mul_t(u1,diff_t(u1,x1))",
    "-0.3332 * diff_t(u1,x1) + -0.1691 * diff_t(n2_t(mul_t(u1,mul_t(u1,u1))),x1) + -0.0524 * u1",
    "0.0393 * diff2_t(n2_t(u1),x1) + -0.3223 * diff_t(u1,x1)",
    "-0.0550 * u1 + -0.3377 * diff_t(u1,x1)",
    "-0.3377 * diff_t(div_t(n2_t(u1),u1),x1) + -0.1358 * n2_t(u1) + -0.0100 * n2_t(diff_t(mul_t(mul_t(mul_t(mul_t(u1,u1),u1),u1),x1))",
    "-0.1717 * diff_t(u1,x1) + -0.1717 * diff_t(u1,x1) + -0.2926 * n2_t(add_t(mul_t(diff_t(div_t(x1,x1),x1),x1),u1)) + 0.0876 * u1",
    "-0.3425 * diff_t(u1,x1) + 0.0420 * diff2_t(u1,x1)",
    "0.0443 * diff2_t(u1,x1) + -0.3618 * diff_t(u1,x1)",
    "0.0428 * u1 + -0.3357 * diff_t(u1,x1)",
    "-0.0230 * n2_t(diff2_t(u1,x1)) + -0.3587 * diff_t(u1,x1)",
    "-0.3334 * diff_t(u1,x1) + 0.0572 * div_t(mul_t(u1,diff2_t(u1,x1)),u1)",
    "-0.0001 * diff_t(div_t(mul_t(x1,u1),u1),x1) + 0.0523 * diff2_t(u1,x1) + -0.3409 * diff_t(u1,x1)",
    "-0.3711 * diff_t(u1,x1) + 0.0121 * diff_t(mul_t(n2_t(x1),n2_t(div_t(mul_t(mul_t(u1,x1),u1),u1))),x1)",
    "-0.3618 * diff_t(mul_t(div_t(u1,u1),u1),x1) + 0.0443 * diff2_t(u1,x1)",
    "-0.3639 * diff_t(u1,x1) + -0.1391 * n2_t(u1)",
    "-0.3353 * diff_t(u1,x1) + 0.0421 * div_t(mul_t(u1,diff2_t(u1,x1)),u1)",
    "-0.3436 * diff_t(u1,x1) + -0.1542 * n2_t(diff_t(mul_t(n2_t(n2_t(u1)),u1),x1))",
]

dprl_cases = [
    "0.0984 * diff2(u1,x1) + -0.5002 * diff(n2(u1),x1)",
    "0.0327 * diff2(mul(u1,n2(x1)),x1) + -0.0003 * diff2(mul(n2(n2(x1)),u1),x1) + -0.0422 * u1",
    "0.1912 * mul(u1,mul(diff(n2(u1),x1),x1))",
    "0.1275 * mul(x1,diff(mul(u1,n2(u1)),x1))",
    "0.0985 * diff2(u1,x1) + -0.5007 * diff(n2(u1),x1) + -0.0001 * diff2(mul(n2(x1),u1),x1)",
    "-0.0009 * u1 + 0.0981 * diff2(u1,x1) + -0.4989 * diff(n2(u1),x1)",
]