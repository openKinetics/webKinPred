// Which kcat methods are allowed for each CSV format detected by the backend.
const kcatMethodsByCsvType = {
  single: ['DLKcat', 'EITLEM', 'UniKP'],
  multi: ['TurNup'],
};

export default kcatMethodsByCsvType;
