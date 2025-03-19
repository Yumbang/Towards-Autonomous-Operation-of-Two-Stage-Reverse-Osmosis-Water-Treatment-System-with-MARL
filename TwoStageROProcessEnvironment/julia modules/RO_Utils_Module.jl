module RO_Utils_Module
    export sherwood, cake_mass, osmo_press, k_CEMT

    eps = 0.36              # Epsilon(ε), the porosity of cake layer [-]
    tau_c = 1 - log(eps^2)  # Tortuosity of the cake layer [-]
    D_b = 1.51e-9           # Hydraulic dispersion coefficient [m2/s]
    D_c = D_b * eps / tau_c # Diffusivity of solute within cake layer [m2/s]
    rho_w = 1023            # rho(ρ), sea water density [g/L] or [kg/m^3]: 35 g/L 25 C 기준

    function sherwood(u, H_local, W, mu)
        nu = mu / rho_w
        dh = 2 * W * H_local / (W + H_local)
        Re = (4 * dh * u) / nu
        Sc = nu / D_b
        Sh = 0.065 * (Re^0.875) * (Sc^0.25)
        k = Sh * D_b / dh
        
        return k
    end

    function cake_mass(v_total, u_total, C_CF_total, i)
        v_crit = 15 / 3600 / 1e3 * (u_total[i] / 0.1)^0.4

        if v_total[i] >= v_crit
            mf_c = C_CF_total[i] * (v_total[i] - v_crit)
        else
            mf_c = 0.0
        end

        return mf_c
    end

    function osmo_press(concentration, temperature)
        osmo_p = 2 / 58.44e3 * concentration * 8.3145e3 * (temperature + 273.15)
        return osmo_p
    end

    function k_CEMT(delta_c, k)
        k_cemt = 1 / (delta_c * (1 / D_c - 1 / D_b) + 1 / k)
        return k_cemt
    end

end
