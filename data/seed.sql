-- Synthetic trade-finance portfolio for Vltava Trade Bank (fictional). No real data.

CREATE TABLE counterparties (
    counterparty_id  INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    country          TEXT NOT NULL,      -- ISO 3166 alpha-2
    role             TEXT NOT NULL,      -- applicant | beneficiary
    risk_rating      TEXT NOT NULL,      -- low | medium | high
    tax_id           TEXT NOT NULL,      -- PII: denied to the agent
    contact_email    TEXT NOT NULL       -- PII: denied to the agent
);

CREATE TABLE letters_of_credit (
    lc_number        TEXT PRIMARY KEY,
    applicant_id     INTEGER NOT NULL REFERENCES counterparties(counterparty_id),
    beneficiary_id   INTEGER NOT NULL REFERENCES counterparties(counterparty_id),
    currency         TEXT NOT NULL,
    amount           REAL NOT NULL,
    issue_date       TEXT NOT NULL,
    expiry_date      TEXT NOT NULL,
    latest_shipment  TEXT NOT NULL,
    status           TEXT NOT NULL,      -- active | expired | cancelled | fully_drawn
    rules            TEXT NOT NULL       -- UCP600 | UCP600+eUCP2.1
);

CREATE TABLE presentations (
    presentation_id  TEXT PRIMARY KEY,
    lc_number        TEXT NOT NULL REFERENCES letters_of_credit(lc_number),
    presented_on     TEXT NOT NULL,
    amount           REAL NOT NULL,
    channel          TEXT NOT NULL,      -- paper | electronic | mixed
    status           TEXT NOT NULL       -- compliant | discrepant | pending | paid
);

CREATE TABLE discrepancies (
    discrepancy_id   INTEGER PRIMARY KEY,
    presentation_id  TEXT NOT NULL REFERENCES presentations(presentation_id),
    code             TEXT NOT NULL,
    description      TEXT NOT NULL,
    waived           INTEGER NOT NULL DEFAULT 0
);

INSERT INTO counterparties VALUES
 (1, 'Moravia Precision Components s.r.o.', 'CZ', 'applicant',   'low',    'CZ27845123', 'treasury@moravia-pc.example'),
 (2, 'Shenzhen Kaida Motion Co., Ltd.',      'CN', 'beneficiary', 'medium', 'CN91440300MA5', 'export@kaida.example'),
 (3, 'Bohemia Glassworks a.s.',              'CZ', 'applicant',   'low',    'CZ45192837', 'finance@bohemia-glass.example'),
 (4, 'Lombardia Forni S.p.A.',               'IT', 'beneficiary', 'low',    'IT03918270164', 'sales@lombardiaforni.example'),
 (5, 'Silesia Agro Trade s.r.o.',            'CZ', 'applicant',   'medium', 'CZ06123987', 'ops@silesia-agro.example'),
 (6, 'Anatolia Tekstil A.S.',                'TR', 'beneficiary', 'high',   'TR4810239985', 'trade@anatolia-tekstil.example'),
 (7, 'Danube Steel Kft.',                    'HU', 'beneficiary', 'medium', 'HU12893456', 'billing@danube-steel.example');

INSERT INTO letters_of_credit VALUES
 ('LC-2026-0142', 1, 2, 'USD', 1250000.00, '2026-06-15', '2026-11-30', '2026-10-31', 'active',      'UCP600+eUCP2.1'),
 ('LC-2026-0187', 3, 4, 'EUR',  480000.00, '2026-08-02', '2026-12-10', '2026-11-15', 'active',      'UCP600'),
 ('LC-2026-0103', 5, 6, 'USD',  310000.00, '2026-03-11', '2026-07-31', '2026-07-10', 'expired',     'UCP600'),
 ('LC-2026-0155', 1, 7, 'EUR',  925000.00, '2026-07-01', '2027-01-31', '2026-12-31', 'active',      'UCP600+eUCP2.1'),
 ('LC-2026-0161', 5, 6, 'USD',  640000.00, '2026-07-20', '2026-10-20', '2026-09-30', 'active',      'UCP600'),
 ('LC-2025-0988', 3, 7, 'EUR',  150000.00, '2025-11-05', '2026-03-31', '2026-03-01', 'fully_drawn', 'UCP600');

INSERT INTO presentations VALUES
 ('PRS-0142-01', 'LC-2026-0142', '2026-09-18', 812400.00, 'mixed',      'discrepant'),
 ('PRS-0103-01', 'LC-2026-0103', '2026-07-22', 305000.00, 'paper',      'paid'),
 ('PRS-0155-01', 'LC-2026-0155', '2026-09-02', 410000.00, 'electronic', 'compliant'),
 ('PRS-0161-01', 'LC-2026-0161', '2026-09-28', 640000.00, 'paper',      'discrepant'),
 ('PRS-0988-01', 'LC-2025-0988', '2026-02-20', 150000.00, 'paper',      'paid'),
 ('PRS-0187-01', 'LC-2026-0187', '2026-10-01', 480000.00, 'paper',      'pending');

INSERT INTO discrepancies VALUES
 (1, 'PRS-0142-01', 'INS-COVER', 'Insurance covers 100% of CIF value instead of 110%', 0),
 (2, 'PRS-0142-01', 'INV-LCREF', 'Commercial invoice does not show the LC number', 0),
 (3, 'PRS-0142-01', 'EBL-PLATF', 'Electronic bill of lading issued on a non-approved platform', 0),
 (4, 'PRS-0161-01', 'LATE-SHIP', 'Shipment effected after the latest shipment date', 0),
 (5, 'PRS-0161-01', 'COO-ISSUER', 'Certificate of origin issued by an entity other than a chamber of commerce', 1);
