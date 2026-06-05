library(AlphaSimR)
library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(purrr)


HER1              <- c(0.10, 0.10, 0.10, 0.40, 0.40, 0.70)
HER2              <- c(0.10, 0.40, 0.70, 0.40, 0.70, 0.70)
TARGET_PREV       <- c(0.10, 0.40)   # target pair prevalence
GEN_COR           <- c(-0.7, 0.7) 
BREED_SCHEME      <- c(0.5, 2)     # sire:dam ratio
SEX_CONT          <- c(0.5, 2)   # sex contribution: female_inf / male_inf
DOM_RATIO         <- 0.5
EPS               <- 1e-6 # clamping
MAXIT             <- 60
SEED              <- 42
N_PAIRS           <- 15000


base <- list(
  h2_m         = 0.40,
  h2_f         = 0.40,
  gen_cor      = 0,
  target_prev  = 0.25,
  mating_scheme= 1,
  dom_r        = 0,
  sex_cont     = 1,
  nQTL         = 100
)

PHI <- function(x) { pnorm(x) }

scenarios <- bind_rows(
  # heritability scenarios
  tibble(
    scen_type = "HER",
    label     = paste0("HER_M", HER1, "_F", HER2),
    h2_m      = HER1,
    h2_f      = HER2,
    gen_cor   = base$gen_cor,
    target_prev  = base$target_prev,
    mating_scheme= base$mating_scheme,
    sex_cont     = base$sex_cont,
    dom_r        = base$dom_r,
    nQTL         = base$nQTL
  ),
  # correlation scenarios
  tibble(
    scen_type = "COR",
    label     = paste0("COR_", GEN_COR),
    h2_m      = base$h2_m,
    h2_f      = base$h2_f,
    gen_cor   = GEN_COR,
    target_prev  = base$target_prev,
    mating_scheme= base$mating_scheme,
    sex_cont     = base$sex_cont,
    dom_r        = base$dom_r,
    nQTL         = base$nQTL
  ),
  # prevalence scenarios
  tibble(
    scen_type = "PREV",
    label     = paste0("PREV_", TARGET_PREV),
    h2_m      = base$h2_m,
    h2_f      = base$h2_f,
    gen_cor   = base$gen_cor,
    target_prev  = TARGET_PREV,
    mating_scheme= base$mating_scheme,
    sex_cont     = base$sex_cont,
    dom_r        = base$dom_r,
    nQTL         = base$nQTL
  ),
  # mating scheme scenarios
  tibble(
    scen_type = "MATE",
    label     = paste0("MATE_", BREED_SCHEME),
    h2_m      = 0.1,
    h2_f      = 0.4,
    gen_cor   = base$gen_cor,
    target_prev  = base$target_prev,
    mating_scheme= BREED_SCHEME,
    sex_cont     = base$sex_cont,
    dom_r        = base$dom_r,
    nQTL         = base$nQTL
  ),
  # sex contribution scenarios
  tibble(
    scen_type = "SEX",
    label     = paste0("SEX_", SEX_CONT),
    h2_m      = 0.1,
    h2_f      = 0.4,
    gen_cor   = base$gen_cor,
    target_prev  = base$target_prev,
    mating_scheme= base$mating_scheme,
    sex_cont     = SEX_CONT,
    dom_r        = base$dom_r,
    nQTL         = base$nQTL
  ),
  # Dominance
  tibble(
    scen_type = "DOM",
    label     = paste0("DOM_", str_pad(DOM_RATIO * 10, 2, pad = "0")),
    h2_m      = base$h2_m,
    h2_f      = base$h2_f,
    gen_cor       = base$gen_cor,
    target_prev   = base$target_prev,
    mating_scheme = base$mating_scheme,
    sex_cont      = base$sex_cont,
    dom_r         = DOM_RATIO,
    nQTL          = base$nQTL      # 10 QTLs for DOM
  )
)


calc_var_components <- function(pop, simparam) {
  GV  <- gv(pop)              # true genetic value
  BV  <- bv(pop, simParam = simparam)              # additive BV
  DDv <- dd(pop, simParam = simparam)              # dominance
  sex <- as.character(pop@sex)
  m <- which(sex == "M")
  f <- which(sex == "F")

  # male fert = trait 1, female fert = trait 2
  VA_m  <- var(BV[m, 1])
  VD_m <- var(DDv[m, 1])
  VG_m  <- var(GV[m, 1])

  VA_f  <- var(BV[f, 2])
  VD_f <- var(DDv[f, 2])
  VG_f  <- var(GV[f, 2])

  data.frame(
    stage = NA_character_,
    VA_m = VA_m, VD_m = VD_m, VG_m = VG_m,
    VA_f = VA_f, VD_f = VD_f, VG_f = VG_f,
    VD_over_VA_m = VD_m / VA_m,
    VD_over_VA_f = VD_f / VA_f,
    VD_over_VG_m = VD_m / VG_m,
    VD_over_VG_f = VD_f / VG_f
  )
}


founder_haplotypes <- runMacs(nInd = 4000, nChr = 10)

########################################################################################

run_one_scenario <- function(founders,
                             h2_m, h2_f,
                             gen_cor,
                             target_prev,
                             mating_scheme,
                             sex_cont,
                             scen_label,
                             rep_id,
                             REP_SEED,
                             N_parents    = 4000,  # total selected parents per gen
                             nGenerations = 5,
                             nCrosses     = 2000,
                             nProgeny     = 15,
                             dom_r,
                             nQTL
) {
  
  compute_sex_infertility <- function(target_pair_inf, sex_contribution_ratio) {
    
    C <- sex_contribution_ratio
    
    a <- C
    b <- -(1 + C)
    c <- target_pair_inf

    # quadradic stuff - for the kids questioning the usefulenss of basic high-school algebra
    disc <- b^2 - 4 * a * c
    if (disc < 0) {
      stop("No solution.")
    }

    root1 <- (-b - sqrt(disc)) / (2 * a)
    root2 <- (-b + sqrt(disc)) / (2 * a)
    
    # Choose the root that lies in [0, 1]
    p_m <- if (!is.na(root1) && root1 >= 0 && root1 <= 1) root1 else root2
    p_f <- C * p_m
    
    list(
      male_infertility   = p_m,
      female_infertility = p_f,
      male_fertility     = 1 - p_m,
      female_fertility   = 1 - p_f
    )
  }

  # Seed for replicate
  set.seed(REP_SEED)
  
  # SimParam + traits (founders are passed - no runMacs)
  simparam <- SimParam$new(founder_haplotypes)
  simparam$setSexes("yes_sys")
  
  # gen corr matrix
  cor_a <- matrix(c(1, gen_cor,
                    gen_cor, 1),
                  ncol = 2, byrow = TRUE)
  
  #var_e <- c(0, dom_r * 1)
  
  Vg_m <- 1
  Vg_f <- 1
  
  meanDD <- dom_r * 2.4
  varDD <- dom_r * 1.6
  
  nQtlPerChr <- rep(nQTL / 10, 10)
  
  simparam$addTraitAD(
    mean   = c(0, 0),
    var    = c(Vg_m, Vg_f),
    meanDD = c(0, meanDD),
    varDD  = c(0, varDD),
    nQtlPerChr = nQtlPerChr)

  simparam$setVarE(h2 = c(h2_m, h2_f))
  
  simparam$addSnpChip(500)
  
  # founders
  pop <- newPop(founder_haplotypes, simparam)
  
  vc0 <- calc_var_components(pop, simparam)
  vc0$stage <- "founders"
  
  # mating_scheme = n_sires/n_dams
  r       <- mating_scheme
  nMale   <- round(N_parents * r / (1 + r))   # sires
  nFemale <- N_parents - nMale                # dams
  
  # burn-in gens
  for (generation in seq_len(nGenerations)) {
    pop <- selectCross(pop      = pop,
                       nFemale  = nFemale,
                       nMale    = nMale,
                       use      = "rand",
                       nCrosses = nCrosses,
                       nProgeny = nProgeny,
                       simParam = simparam)
  }
  
  pop <- setPheno(pop, simParam=simparam)

  GV    <- gv(pop)
  BV    <- bv(pop, simParam=simparam)
  PHENO <- pheno(pop)
  
  vc1 <- calc_var_components(pop, simparam)
  vc1$stage <- "post_burnin"

  var_components <- dplyr::bind_rows(vc0, vc1)
  
  sex_vals <- as.character(pop@sex)
  
  sire_indices_master <- which(sex_vals == "M")
  dam_indices_master  <- which(sex_vals == "F")
  
  N_SIRES_MASTER <- length(sire_indices_master)
  N_DAMS_MASTER  <- length(dam_indices_master)
  
  ph_m <- PHENO[sire_indices_master, 1]
  ph_f <- PHENO[dam_indices_master, 2]
  a    <- GV[sire_indices_master, 1]
  b    <- GV[dam_indices_master, 2]
  
  sire_ids_master <- pop@id[sire_indices_master]
  dam_ids_master  <- pop@id[dam_indices_master]
  
  # sex-specific infertility from target infertility
  inf_res <- compute_sex_infertility(
    target_pair_inf       = target_prev,
    sex_contribution_ratio = sex_cont
  )
  
  male_inf_prev   <- inf_res$male_infertility
  female_inf_prev <- inf_res$female_infertility
  
  n_m <- length(ph_m)
  n_f <- length(ph_f)
  
  nInf_m <- round(male_inf_prev   * n_m)
  nInf_f <- round(female_inf_prev * n_f)
  
  ph_m_clean <- as.vector(c(ph_m), mode = "numeric") 
  ord_m <- sort.list(ph_m_clean, decreasing = F)
  
  ph_f_clean <- as.vector(c(ph_f), mode = "numeric")
  ord_f <- sort.list(ph_f_clean, decreasing = F)
  
  inf_m <- integer(n_m)
  inf_f <- integer(n_f)
  
  inf_m[ord_m[1:nInf_m]] <- 1L
  inf_f[ord_f[1:nInf_f]] <- 1L
  
  # Define unique sires/dams and build 15k pairs
  if (r >= 1) {
    # more sires
    N_SIRES_UNIQ_TARGET <- min(N_PAIRS, N_SIRES_MASTER)
    N_DAMS_UNIQ_TARGET  <- min(ceiling(N_PAIRS / r), N_DAMS_MASTER)
  } else {
    # more dams
    N_SIRES_UNIQ_TARGET <- min(ceiling(N_PAIRS * r), N_SIRES_MASTER)
    N_DAMS_UNIQ_TARGET  <- min(N_PAIRS, N_DAMS_MASTER)
  }
  
  if (N_SIRES_UNIQ_TARGET == 0L || N_DAMS_UNIQ_TARGET == 0L) {
    stop("Insufficient sires or dams")
  }
  
  # indices
  uniq_sire_ix <- sample.int(N_SIRES_MASTER, N_SIRES_UNIQ_TARGET, replace = FALSE)
  uniq_dam_ix  <- sample.int(N_DAMS_MASTER,  N_DAMS_UNIQ_TARGET,  replace = FALSE)
  
  # pair-level ind
  if (r >= 1) {
    # sires mostly unique
    if (N_SIRES_UNIQ_TARGET >= N_PAIRS) {
      pair_sire_ix <- sample(uniq_sire_ix, N_PAIRS, replace = FALSE)
    } else {
      pair_sire_ix <- sample(uniq_sire_ix, N_PAIRS, replace = TRUE)
    }
    pair_dam_ix <- sample(rep(uniq_dam_ix, length.out = N_PAIRS))
  } else {
    # dams mostly unique
    if (N_DAMS_UNIQ_TARGET >= N_PAIRS) {
      pair_dam_ix <- sample(uniq_dam_ix, N_PAIRS, replace = FALSE)
    } else {
      pair_dam_ix <- sample(uniq_dam_ix, N_PAIRS, replace = TRUE)
    }
    pair_sire_ix <- sample(rep(uniq_sire_ix, length.out = N_PAIRS))
  }
  
  # phenotypes
  sire_pheno_vec <- ph_m[pair_sire_ix]
  dam_pheno_vec  <- ph_f[pair_dam_ix]
  
  pair_sire_inf <- inf_m[pair_sire_ix]
  pair_dam_inf  <- inf_f[pair_dam_ix]
  
  # infertility rule
  BinPheno <- as.integer(pair_sire_inf | pair_dam_inf)  # 0 = fertile, 1 = infertile
  
  sire_bv_vec <- a[pair_sire_ix]
  dam_bv_vec  <- b[pair_dam_ix]
  
  # Global IDs
  sire_id_vec <- sire_ids_master[pair_sire_ix]
  dam_id_vec  <- dam_ids_master[pair_dam_ix]
  
  cross_ids <- paste0("X", stringr::str_pad(seq_len(N_PAIRS), 6, side = "left", pad = "0"))
  
  SimPhen <- data.frame(
    Sire        = paste0("AN", sire_id_vec),
    Dam         = paste0("AN", dam_id_vec),
    CrossID     = cross_ids,
    SirePheno   = sire_pheno_vec,
    DamPheno    = dam_pheno_vec,
    SireBV      = sire_bv_vec,
    DamBV       = dam_bv_vec,
    SireInf     = pair_sire_inf,
    DamInf      = pair_dam_inf,
    BinPheno    = BinPheno,
    stringsAsFactors = FALSE
  )
  
  SNPs <- pullSnpGeno(pop = pop, snpChip = 1, simParam = simparam)
  row.names(SNPs) <- paste0("AN", row.names(SNPs))
  
  QTLs <- colnames(pullQtlGeno(pop = pop, simParam = simparam))
  
  phen_fname <- sprintf("SimPhen_%s_R%02d_15k.csv", scen_label, rep_id)
  snp_fname  <- sprintf("SimFeat_%s_R%02d_15k.csv", scen_label, rep_id)
  qtl_fname  <- sprintf("SimQTL_%s_R%02d_15k.csv", scen_label, rep_id)
  
  readr::write_csv(SimPhen, phen_fname)
  write.csv(SNPs, snp_fname, quote = FALSE)
  write.csv(QTLs, qtl_fname)
  
  # realized prevalence
  achieved_prev <- mean(BinPheno == 1L)
  
  invisible(list(
    SimPhen        = SimPhen,
    SNPs           = SNPs,
    target_prev    = target_prev,
    achieved_prev  = achieved_prev,
    male_inf_prev  = male_inf_prev,
    female_inf_prev= female_inf_prev,
    nMale          = nMale,
    nFemale        = nFemale,
    N_pairs        = N_PAIRS,
    var_components = var_components
  ))
}


N_REP <- 5

var_log <- vector("list", length = nrow(scenarios) * N_REP)
k <- 1

for (i in seq_len(nrow(scenarios))) {
  scen <- scenarios[i, ]
  
  for (rep_id in 1:N_REP) {
    message("Running scenario ", scen$label, " replicate ", rep_id)
    
    out <- run_one_scenario(
      founders      = founder_haplotypes,
      h2_m          = scen$h2_m,
      h2_f          = scen$h2_f,
      gen_cor       = scen$gen_cor,
      target_prev   = scen$target_prev,
      mating_scheme = scen$mating_scheme,
      sex_cont      = scen$sex_cont,
      scen_label    = scen$label,
      nQTL          = scen$nQTL,
      dom_r         = scen$dom_r,
      rep_id        = rep_id,
      REP_SEED      = rep_id + 283
    )
    
    var_log[[k]] <- cbind.data.frame(
      scenario = scen$label,
      rep      = rep_id,
      out$var_components)
      
    k <- k + 1
  }
}

var_log <- dplyr::bind_rows(var_log)
readr::write_csv(var_log, "dom_realized_variances.csv")
