// Helpful discussion at https://github.com/farr/SelectionExample/blob/master/Gaussian.ipynb

functions {

  // transform flux into luminosity
  // for now this is WRONG
  real transform(real x, real z) {
    return x * 4* pi() * (z+1)^2 ; 
  }
  
  
  // variable Poisson intensity
  vector dNdV(vector z, real r0, real rise, real decay, real peak) {

    int N = num_elements(z);

    vector[N] bottom;
    vector[N] top;
    vector[N] frac;

    top = r0 * (1.0 + rise*z);
    frac = z/peak;
    for (n in 1:N) {

      bottom[n] = 1+frac[n]^decay;

    }

    return top ./ bottom;
  }



  vector dVdz(vector z) {

    return 4 * pi() * (z+1) .* (z+1);

  }

  real dVdz_int(real z) {

    return 4 * pi() * (z+1) * (z+1);

  }
  
 real dNdV_int(real z, real r0, real rise, real decay, real peak) {
    real bottom;
    real  top;
    

    top = r0 * (1.0 + rise*z);
    bottom =1+(z/peak)^decay;
    

    return top / bottom;
  }

  // Integrand of the rate
  real[] N_integrand(real z, real[] state, real[] params, real[] x_r, int[] x_i) {
    
    real r0;
    real rise;
    real decay;
    real peak;
    
    real dstatedz[1];
    
    r0 = params[1];
    rise = params[2];
    decay = params[3];
    peak = params[4];
    
    
    dstatedz[1] = dNdV_int(z, r0, rise, decay, peak) * dVdz_int(z);
    
    return dstatedz;
  }
  
}

data {

  // luminosity
  int<lower=1> N;
  vector[N] L_obs;
  real sigma_L;

  // redshift
  vector[N] z_obs;
  real<lower=0> z_max;

  real<lower=0> rise; // formation fall off
  real<lower=0> decay; // formation fall off
  real<lower=0, upper=z_max> peak; // formation fall off
  
  // marginalization
  int<lower=0> N_margin;
  real boundary;

  // gen
  int N_model;
  vector[N_model] zmodel;
  vector[N_model] Lmodel;
  
}

transformed data {

  // technical varaibales required for the integration
  real x_r[0];
  int x_i[0];
  real zout[1];
  vector[N_margin] log_gamma;

  zout[1] = z_max;

  // precompute for speed
  for (n in 1:N_margin) {
    log_gamma[n] = lgamma(n+1);
  }

}

parameters {

  // Phi(L) and dNdZ(z) parameters
  real<lower=0> alpha; // phi index 
  real<lower=0> Lstar; // phi lower bound
  real<lower=0> r0; // local rate
  /* real<lower=0> rise; // formation fall off */
  /* real<lower=0> decay; // formation fall off */
  /* real<lower=0, upper=z_max> peak; // formation fall off */

  // latent luminosities
  vector<lower=Lstar>[N] L_latent;

  // auxilliary population
  vector<lower=0,upper=boundary>[N_margin] F_tilde;
  vector<lower=Lstar>[N_margin] L_tilde_latent;
  vector<lower=0,upper=z_max>[N_margin] z_tilde_latent;

}

transformed parameters {
  //    vector<lower=0>[N_margin] L_tilde_transformed;
  
  // transform the flux into a luminosity
  /* for (n in 1:N_margin) { */
  /*   L_tilde_transformed[n] = transform(F_tilde[n], z_tilde_latent[n]);     */
  /* } */
  

}

model {
  real sum_log_prob_tilde;
  vector[N_margin + 1] log_prob_margin;  

  // setup for the integral
  real Lambda; // this is total Lambda!
  
  real params[4];
  real integration_result[1,1];
  real state0[1];

  vector[N_margin] dz_tilde;

  // positive definite priors for the intensity
  r0 ~ lognormal(log(1.), 1.0);
  //  rise ~ normal(1.0, 1.0);
  //  decay ~ normal(2.0, 2.0);
  // peak ~lognormal(0.,1.);

  
  // priors for distributions
  alpha ~ normal(1.0, 1.0);
  Lstar ~ lognormal(log(1.0), 1.0);
  
  // Measurement model for observed objects
  L_latent ~  pareto(Lstar, alpha);
  L_obs ~ lognormal(log(L_latent), sigma_L);

  // add the differential of the inhomogeneous process on

  // remember that we are looking for dN/dz = (dN/dV) * (dV/dz)
  
  target += log(dNdV(z_obs, r0, rise, decay, peak) .* dVdz(z_obs));
  
  
  

  // Measurement model for auxiliary objects  
  L_tilde_latent ~  pareto(Lstar, alpha);

  // normalize with the offset distributions
  target += uniform_lpdf(z_tilde_latent| 0, z_max);
  target += uniform_lpdf(F_tilde |0,  boundary);

  // these norms give me a headache
  // I'm not really sure if they are correct

  
  
  log_prob_margin[1] = 0;
  sum_log_prob_tilde = log_prob_margin[1];

  // it is vectorized so precompute it here.
  dz_tilde = log(dNdV(z_tilde_latent, r0, rise, decay, peak) .* dVdz(z_tilde_latent));

  for (n in 1:N_margin) {

    real L_tilde_transformed = transform(F_tilde[n], z_tilde_latent[n]);    

    sum_log_prob_tilde += dz_tilde[n]
      - uniform_lpdf(F_tilde[n] | 0, boundary) // remove excess prob
      - uniform_lpdf(z_tilde_latent[n]| 0, z_max) // remove excess prob
      + lognormal_lpdf(L_tilde_transformed | log(L_tilde_latent[n]), sigma_L);

      
     
    // the full sum plus the cumulative part.
    // note that that the log gamma is precomputed
    
    log_prob_margin[n + 1] =  -log_gamma[n] + sum_log_prob_tilde;
  }

  target += log_sum_exp(log_prob_margin);


  // Poisson normalization for the integral rate

  params[1] = r0;
  params[2] = rise;
  params[3] = decay;
  params[4] = peak;
  
  state0[1] = 0.0;
  
  // integrate the dN/dz to get the normalizing constant for given r0 and alpha
  integration_result = integrate_ode_rk45(N_integrand, state0, 0.0, zout, params, x_r, x_i);
  Lambda = integration_result[1,1];

  // (Distinguishiable) Poisson process model
  target += - Lambda;

}

generated quantities {
  real dNdV_model[N_model];
  real phi_model[N_model];
  
  for (i in 1:N_model) {
    dNdV_model[i] = dNdV_int(zmodel[i], r0, rise, decay, peak);
    phi_model[i] = exp(pareto_lpdf(Lmodel[i]| Lstar, alpha));
  }
}
