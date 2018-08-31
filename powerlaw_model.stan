// Helpful discussion at https://github.com/farr/SelectionExample/blob/master/Gaussian.ipynb

functions {

  // transform flux into luminosity
  real transform(real x, real z) {
    return x * (z+1)^2 ; 
  }
  
  
  // variable Poisson intensity
  real dNdz(real z, real r0, real index) {
    return r0 * (1.0 + z)^(index);
  }
  
  
  real[] N_integrand(real z, real[] state, real[] params, real[] x_r, int[] x_i) {
    
    real r0;
    real index;
    
    real dstatedz[1];
    
    r0 = params[1];
    index = params[2];
    
    
    dstatedz[1] = dNdz(z, r0, index);
    
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
  
  for (n in 1:N_margin) {
    log_gamma[n] = lgamma(n+1);
  }

}

parameters {

  
  real<lower=0> alpha;
  real<lower=0> Lstar;  
  real<lower=0> r0;
  real index;

  
  vector<lower=Lstar>[N] L_latent;


  vector<lower=0,upper=boundary>[N_margin] F_tilde;
  vector<lower=Lstar>[N_margin] L_tilde_latent;
  vector<lower=0,upper=z_max>[N_margin] z_tilde_latent;

}

transformed parameters {
    vector<lower=0>[N_margin] L_tilde_transformed;
  
  // transform the flux into a luminosity
  for (n in 1:N_margin) {
    L_tilde_transformed[n] = transform(F_tilde[n], z_tilde_latent[n]);    
  }
  

}

model {
  real sum_log_prob_tilde;
  vector[N_margin + 1] log_prob_margin;  

  // setup for the integral
  real Lambda;
  real params[2];
  real integration_result[1,1];
  real state0[1];




  // positive definite priors for the intensity
  r0 ~ lognormal(log(100.0), 1.0);
  index ~ normal(-1.0, 1.0);
  

  // priors for distributions
  alpha ~ normal(1.0, 1.0);
  Lstar ~ lognormal(log(1.0), 1.0);
  
  // Measurement model for observed objects
  L_latent ~  pareto(Lstar, alpha);
  L_obs ~ lognormal(log(L_latent), sigma_L);

  // add the differential of the inhomogeneous process on
  for (n in 1:N) {
    target += log(dNdz(z_obs[n], r0, index));
  }
  
  // (Distinguishiable) Poisson process model

  // Measurement model for auxiliary objects  
  L_tilde_latent ~  pareto(Lstar, alpha);
  
  target += uniform_lpdf(z_tilde_latent| 0, z_max);
  target += uniform_lpdf(F_tilde |0,  boundary);

  
  log_prob_margin[1] = 0;
  sum_log_prob_tilde = log_prob_margin[1];


  for (n in 1:N_margin) {
    
    sum_log_prob_tilde += log(dNdz(z_tilde_latent[n], r0, index))
      - uniform_lpdf(F_tilde[n] | 0, boundary)
      - uniform_lpdf(z_tilde_latent[n]| 0, z_max)
      + lognormal_lpdf(L_tilde_transformed[n] | log(L_tilde_latent[n]), sigma_L);

      
     
    // the full sum plus the cumulative part.
    // note that that the log gamma is precomputed
    
    log_prob_margin[n + 1] =  -log_gamma[n] + sum_log_prob_tilde;
  }

  target += log_sum_exp(log_prob_margin);


  // Poisson normalization for the integral rate

  params[1] = r0;
  params[2] = index;
 
  state0[1] = 0.0;
  
  // integrate the dN/dz to get the normalizing constant for given r0 and alpha
  integration_result = integrate_ode_rk45(N_integrand, state0, 0.0, zout, params, x_r, x_i);
  Lambda = integration_result[1,1];

  target += - Lambda;

}

generated quantities {
  real dNdz_model[N_model];
  real phi_model[N_model];
  
  for (i in 1:N_model) {
    dNdz_model[i] = dNdz(zmodel[i], r0, index);
    phi_model[i] = exp(pareto_lpdf(Lmodel[i]| Lstar, alpha));
  }
}
