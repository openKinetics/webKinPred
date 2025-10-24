// Descriptions and citations shown to users.
// Keep purely static data here to avoid re-creating on each render.

const methodDetails = {
  "KinForm-H": {
    description: 'Placeholder description for KinForm-H method.',
    authors: 'Author A, Author B',
    publicationTitle: 'Placeholder Title for KinForm-H',
    citationUrl: 'placeholder-url-for-kinform-h',
    repoUrl: 'https://github.com/Digital-Metabolic-Twin-Centre/KinForm',
  },
  "KinForm-L": {
    description: 'Placeholder description for KinForm-L method.',
    authors: 'Author C, Author D',
    publicationTitle: 'Placeholder Title for KinForm-L',
    citationUrl: 'placeholder-url-for-kinform-l',
    repoUrl: 'https://github.com/Digital-Metabolic-Twin-Centre/KinForm',
  },
  UniKP: {
    description: 'Predicts kcat or KM for a reaction given protein sequence + substrate.',
    authors: 'Han Yu, Huaxiang Deng, Jiahui He, Jay D. Keasling & Xiaozhou Luo',
    publicationTitle: 'UniKP: a unified framework for the prediction of enzyme kinetic parameters',
    citationUrl: 'https://www.nature.com/articles/s41467-023-44113-1',
    repoUrl: 'https://github.com/Luo-SynBioLab/UniKP',
  },
  DLKcat: {
    description: 'Predicts kcat for a reaction given protein sequence + substrate.',
    authors: 'Feiran Li, Le Yuan, Hongzhong Lu, Gang Li, Yu Chen, Martin K. M. Engqvist, Eduard J. Kerkhoven & Jens Nielsen',
    publicationTitle: 'Deep learning-based kcat prediction enables improved enzyme-constrained model reconstruction',
    citationUrl: 'https://www.nature.com/articles/s41929-022-00798-z',
    repoUrl: 'https://github.com/SysBioChalmers/DLKcat',
    moreInfo: '',
  },
  TurNup: {
    description: 'Predicts kcat for each reaction given protein sequence + list of substrates + list of products.',
    authors: 'Alexander Kroll, Yvan Rousset, Xiao-Pan Hu, Nina A. Liebrand & Martin J. Lercher',
    publicationTitle: 'Turnover number predictions for kinetically uncharacterised enzymes using machine and deep learning',
    citationUrl: 'https://www.nature.com/articles/s41467-023-39840-4',
    repoUrl: 'https://github.com/AlexanderKroll/Kcat_prediction',
    moreInfo: 'Recommended for natural reactions of wild-type enzymes.',
  },
  EITLEM: {
    description: 'Predicts kcat or KM for a reaction given protein sequence + substrate.',
    authors: 'Xiaowei Shen, Ziheng Cui, Jianyu Long, Shiding Zhang, Biqiang Chen, Tianwei Tan',
    publicationTitle: 'EITLEM-Kinetics: A deep-learning framework for kinetic parameter prediction of mutant enzymes',
    citationUrl: 'https://www.sciencedirect.com/science/article/pii/S2667109324002665',
    repoUrl: 'https://github.com/XvesS/EITLEM-Kinetics',
    moreInfo: 'Recommended for mutants.',
  },
};

export default methodDetails;
