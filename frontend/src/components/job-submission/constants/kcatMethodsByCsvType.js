// Which kcat methods are allowed for each CSV format detected by the backend.
const kcatMethodsByCsvType = {
  single: ['DLKcat', 'EITLEM', 'UniKP', 'KinForm-H', 'KinForm-L'],
  multi: ['TurNup'],
};

export default kcatMethodsByCsvType;
