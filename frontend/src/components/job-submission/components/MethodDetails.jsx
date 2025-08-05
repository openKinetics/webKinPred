import React from 'react';
import PropTypes from 'prop-types';
import { Alert } from 'react-bootstrap';
import methodDetails from '../constants/methodDetails';

export default function MethodDetails({ methodKey, citationOnly = false }) {
  const method = methodDetails[methodKey];
  if (!method) return null;

  return (
    <Alert variant="info" className="mt-3">
      {!citationOnly && <p>{method.description}</p>}
      {method.publicationTitle && method.citationUrl && (
        <p>
          <strong>Publication: </strong>
          <a href={method.citationUrl} target="_blank" rel="noopener noreferrer">
            {method.publicationTitle}
          </a>
        </p>
      )}
      {!citationOnly && method.moreInfo && (
        <p><strong>Note: </strong>{method.moreInfo}</p>
      )}
    </Alert>
  );
}

MethodDetails.propTypes = {
  methodKey: PropTypes.string,
  citationOnly: PropTypes.bool,
};
