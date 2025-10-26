// Which kcat methods are allowed for each CSV format detected by the backend.
const kcatMethodsByCsvType = {
  single: [ 'KinForm-H', 'KinForm-L','DLKcat', 'EITLEM', 'UniKP'],
  multi: ['TurNup'],
};

export default kcatMethodsByCsvType;
