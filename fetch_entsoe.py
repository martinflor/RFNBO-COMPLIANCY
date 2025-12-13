from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session
import requests
import xml.etree.ElementTree as ET
import time
import logging
import zipfile
import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Union, Dict, Any
import os
import re
import pytz

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENTSOE_API_KEY = os.getenv('ENTSOE_API_KEY', "e094c8aa-00ae-4063-8a78-1d712d2ea774")
BASE_URL = 'https://web-api.tp.entsoe.eu/api'

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504, 429],
    allowed_methods=["GET"]
)

# Create a session with retry strategy
session = requests.Session()
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

BIDDING_ZONES: Dict[str, str] = {
    'Austria':                    '10YAT-APG------L',
    'Belgium':                    '10YBE----------2',
    'Bulgaria':                   '10YCA-BULGARIA-R',
    'Croatia':                    '10YHR-HEP------M',
    'Czech Republic':             '10YCZ-CEPS-----N',
    'Denmark 1':                  '10YDK-1--------W',
    'Denmark 2':                  '10YDK-2--------M',
    'Estonia':                    '10Y1001A1001A39I',
    'Finland':                    '10YFI-1--------U',
    'France':                     '10YFR-RTE------C',
    'Germany (DE/LU)':            '10YCB-GERMANY--8',
    'Germany 50Hertz':            '10Y1001A1001A63L',  # 50Hertz area
    'Germany Amprion':            '10Y1001A1001A82H',  # Amprion area
    'Germany TenneT':             '10Y1001A1001A83F',  # TenneT area
    'Germany TransnetBW':         '10Y1001A1001A87K',  # TransnetBW area
    'Greece':                     '10YGR-HTSO-----Y',
    'Hungary':                    '10YHU-MAVIR----U',
    'Ireland':                    '10YIE-1001A00010',
    'Italy Centro Nord':          '10Y1001A1001A70O',
    'Italy Centro Sud':           '10Y1001A1001A71M',
    'Lithuania':                  '10YLT-1001A0008Q',
    'Malta':                      '10Y1001A1001A877',
    'Netherlands':                '10YNL----------L',
    'Norway 1':                   '10YNO-1--------2',
    'Norway 2':                   '10YNO-2--------T',
    'Norway 3':                   '10YNO-3--------J',
    'Norway 4':                   '10YNO-4--------9',
    'Norway 5':                   '10Y1001A1001A48H',
    'Poland':                     '10YPL-AREA-----S',
    'Romania':                    '10YRO-TEL------P',
    'Serbia':                     '10YCS-SERBIATSOV',
    'Slovakia':                   '10YSK-SEPS-----K',
    'Slovenia':                   '10YSI-ELES-----O',
    'Spain':                      '10YES-REE------0',
    'Sweden 1':                   '10Y1001A1001A44P',
    'Sweden 2':                   '10Y1001A1001A45N',
    'Sweden 3':                   '10Y1001A1001A46L',
    'Sweden 4':                   '10Y1001A1001A47J',
    'Switzerland':                '10Y1001A1001A68B',
}

# Timezone mappings for each country (using pytz timezone names)
COUNTRY_TIMEZONES: Dict[str, str] = {
    'Austria': 'Europe/Vienna',
    'Belgium': 'Europe/Brussels',
    'Bulgaria': 'Europe/Sofia',
    'Croatia': 'Europe/Zagreb',
    'Czech Republic': 'Europe/Prague',
    'Denmark 1': 'Europe/Copenhagen',
    'Denmark 2': 'Europe/Copenhagen',
    'Estonia': 'Europe/Tallinn',
    'Finland': 'Europe/Helsinki',
    'France': 'Europe/Paris',
    'Germany (DE/LU)': 'Europe/Berlin',
    'Greece': 'Europe/Athens',
    'Hungary': 'Europe/Budapest',
    'Ireland': 'Europe/Dublin',
    'Italy Centro Nord': 'Europe/Rome',
    'Italy Centro Sud': 'Europe/Rome',
    'Lithuania': 'Europe/Vilnius',
    'Malta': 'Europe/Malta',
    'Netherlands': 'Europe/Amsterdam',
    'Norway 1': 'Europe/Oslo',
    'Norway 2': 'Europe/Oslo',
    'Norway 3': 'Europe/Oslo',
    'Norway 4': 'Europe/Oslo',
    'Norway 5': 'Europe/Oslo',
    'Poland': 'Europe/Warsaw',
    'Romania': 'Europe/Bucharest',
    'Serbia': 'Europe/Belgrade',
    'Slovakia': 'Europe/Bratislava',
    'Slovenia': 'Europe/Ljubljana',
    'Spain': 'Europe/Madrid',
    'Sweden 1': 'Europe/Stockholm',
    'Sweden 2': 'Europe/Stockholm',
    'Sweden 3': 'Europe/Stockholm',
    'Sweden 4': 'Europe/Stockholm',
    'Switzerland': 'Europe/Zurich',
}

# Mapping from frontend country names to backend country names
FRONTEND_TO_BACKEND_COUNTRIES: Dict[str, str] = {
    'Germany': 'Germany (DE/LU)',  # Use the main German bidding zone for day-ahead prices
    'Italy': 'Italy Centro Nord',  # Default to Centro Nord, could be made configurable
    'Denmark': 'Denmark 1',         # Default to Denmark 1, could be made configurable
    'Sweden': 'Sweden 1',           # Default to Sweden 1, could be made configurable
    'Norway': 'Norway 1',           # Default to Norway 1, could be made configurable
}

def get_backend_country_name(frontend_country: str) -> str:
    """
    Convert frontend country name to backend country name.
    """
    return FRONTEND_TO_BACKEND_COUNTRIES.get(frontend_country, frontend_country)

def get_frontend_country_name(backend_country: str) -> str:
    """
    Convert backend country name to frontend country name.
    """
    # Create reverse mapping
    backend_to_frontend = {v: k for k, v in FRONTEND_TO_BACKEND_COUNTRIES.items()}
    return backend_to_frontend.get(backend_country, backend_country)

def get_timezone_offset_hours(zone_name: str, target_date: datetime) -> int:
    """
    Get the timezone offset in hours for a given country and date.
    This handles daylight saving time automatically.
    """
    timezone_name = COUNTRY_TIMEZONES.get(zone_name, 'Europe/Brussels')
    tz = pytz.timezone(timezone_name)
    
    # Localize the target date to the timezone
    localized_date = tz.localize(target_date)
    
    # Get the UTC offset in hours
    offset_hours = int(localized_date.utcoffset().total_seconds() / 3600)
    
    return offset_hours

def _check_entsoe_error(xml_text: str) -> None:
    """
    Check if the XML response contains an ENTSO-E error message.
    """
    try:
        root = ET.fromstring(xml_text)
        # Look for common error elements in ENTSO-E responses
        # Try different possible error element names
        error_elements = []
        
        # Common ENTSO-E error elements
        error_elements.extend(root.findall(".//*[local-name()='Reason']"))
        error_elements.extend(root.findall(".//*[local-name()='text']"))
        error_elements.extend(root.findall(".//*[local-name()='error']"))
        error_elements.extend(root.findall(".//*[local-name()='Error']"))
        error_elements.extend(root.findall(".//*[local-name()='Message']"))
        
        # Also check for specific ENTSO-E error codes
        error_elements.extend(root.findall(".//*[local-name()='code']"))
        error_elements.extend(root.findall(".//*[local-name()='Code']"))
        
        if error_elements:
            error_msg = " ".join([elem.text for elem in error_elements if elem.text])
            if error_msg and any(keyword in error_msg.lower() for keyword in ['error', 'invalid', 'not found', 'unavailable', 'no data']):
                logger.error(f"ENTSO-E API returned error: {error_msg}")
                raise ValueError(f"ENTSO-E API error: {error_msg}")
                
    except ET.ParseError:
        # If we can't parse the XML, it's not an XML error message
        pass
    except Exception as e:
        # Don't raise here, just log the error
        logger.debug(f"Error checking for ENTSO-E errors: {str(e)}")

def _extract_xml_from_zip(zip_data: bytes) -> str:
    """
    Extract XML content from a ZIP file response.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_file:
            # Look for XML files in the ZIP
            xml_files = [f for f in zip_file.namelist() if f.endswith('.xml')]
            
            if not xml_files:
                raise ValueError("No XML files found in ZIP response")
            
            # Use the first XML file found
            xml_filename = xml_files[0]
            logger.info(f"Extracting XML from ZIP file: {xml_filename}")
            
            with zip_file.open(xml_filename) as xml_file:
                xml_content = xml_file.read().decode('utf-8')
                return xml_content
                
    except zipfile.BadZipFile:
        raise ValueError("Response is not a valid ZIP file")
    except Exception as e:
        raise ValueError(f"Failed to extract XML from ZIP: {str(e)}")

def _query_entsoe(params: Dict[str, Any]) -> str:
    """
    Send the GET request and return raw XML text (raises for non-200).
    """
    params = {**params, "securityToken": ENTSOE_API_KEY}
    
    # Log the request (without the security token for privacy)
    safe_params = {k: v for k, v in params.items() if k != "securityToken"}
    logger.info(f"Making ENTSO-E API request with params: {safe_params}")
    
    resp = session.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    
    logger.info(f"Received response with status {resp.status_code}, content-type: {resp.headers.get('content-type', 'unknown')}")
    
    content_type = resp.headers.get('content-type', '').lower()
    
    # Handle ZIP responses
    if 'zip' in content_type or 'application/zip' in content_type:
        logger.info("Detected ZIP response, extracting XML content")
        try:
            xml_content = _extract_xml_from_zip(resp.content)
            logger.info("Successfully extracted XML from ZIP")
            return xml_content
        except Exception as e:
            logger.error(f"Failed to extract XML from ZIP: {str(e)}")
            raise ValueError(f"Failed to extract XML from ZIP response: {str(e)}")
    
    # Handle direct XML responses
    elif 'xml' in content_type or 'text' in content_type:
        # Check if response is actually XML
        if 'xml' not in content_type and 'text' in content_type:
            # Log the first 500 characters of the response for debugging
            response_preview = resp.text[:500]
            logger.error(f"API returned non-XML response. Content-Type: {content_type}. Response preview: {response_preview}")
            raise ValueError(f"API returned non-XML response. Content-Type: {content_type}. Response preview: {response_preview}")
        
        # Check for ENTSO-E error messages in the response
        _check_entsoe_error(resp.text)
        
        # Additional check: try to parse as XML to catch malformed XML early
        try:
            ET.fromstring(resp.text)
            logger.info("Successfully parsed XML response")
        except ET.ParseError as e:
            # Log the first 500 characters of the malformed response
            response_preview = resp.text[:500]
            logger.error(f"API returned malformed XML. Parse error: {str(e)}. Response preview: {response_preview}")
            raise ValueError(f"API returned malformed XML. Parse error: {str(e)}. Response preview: {response_preview}")
        
        return resp.text
    else:
        # Unknown content type, try to handle as XML anyway
        logger.warning(f"Unknown content type: {content_type}, attempting to parse as XML")
        try:
            ET.fromstring(resp.text)
            logger.info("Successfully parsed response as XML despite unknown content type")
            return resp.text
        except ET.ParseError as e:
            response_preview = resp.text[:500]
            logger.error(f"Failed to parse response as XML. Parse error: {str(e)}. Response preview: {response_preview}")
            raise ValueError(f"Failed to parse response as XML. Parse error: {str(e)}. Response preview: {response_preview}")

def _xml_timeseries_to_df(xml_text: str, value_tag: str = "quantity") -> pd.DataFrame:
    """
    Parse ANY ENTSO-E TimeSeries document into a long DataFrame with:
        timestamp · value · psr_type · business_type · resolution_minutes
    `value_tag` is usually 'quantity' (MWh) but can be 'price.amount'.
    """
    logger.info(f"Starting XML parsing with value_tag: {value_tag}")
    _check_entsoe_error(xml_text)
    logger.info("No ENTSO-E errors found in XML")
    ns = {"ns": ET.fromstring(xml_text).tag.split("}")[0].strip("{")}
    logger.info(f"Using namespace: {ns}")
    root = ET.fromstring(xml_text)
    
    # Log the root element to see what type of document we received
    logger.info(f"Root element: {root.tag}")
    if 'acknowledgement' in root.tag.lower():
        logger.warning("Received acknowledgement document instead of data - this usually means no data available for the requested date/zone")
        # Try to extract any error message from the acknowledgement
        for elem in root.findall(".//*"):
            if elem.text and elem.text.strip():
                logger.info(f"Document content: {elem.tag} = {elem.text}")
        # Return empty DataFrame for acknowledgement documents
        logger.info("Returning empty DataFrame for acknowledgement document")
        return pd.DataFrame()
    
    rows = []

    for ts in root.findall(".//ns:TimeSeries", ns):
        psr = ts.find("ns:mktPSRType/ns:psrType", ns)
        business = ts.find("ns:businessType", ns)
        for period in ts.findall("ns:Period", ns):
            start = pd.Timestamp(period.find("ns:timeInterval/ns:start", ns).text)
            # Get resolution (e.g., PT15M, PT60M)
            res_elem = period.find("ns:resolution", ns)
            if res_elem is not None:
                res_str = res_elem.text
                match = re.match(r"PT(\d+)M", res_str)
                if match:
                    minutes = int(match.group(1))
                    resolution = pd.Timedelta(minutes=minutes)
                else:
                    # fallback to 1 hour if not found
                    minutes = 60
                    resolution = pd.Timedelta(hours=1)
            else:
                # fallback to 1 hour if not found
                minutes = 60
                resolution = pd.Timedelta(hours=1)
            for point in period.findall("ns:Point", ns):
                position_elem = point.find("ns:position", ns)
                value_elem = point.find(f"ns:{value_tag}", ns)
                if position_elem is None or position_elem.text is None:
                    continue
                if value_elem is None or value_elem.text is None:
                    continue
                try:
                    position = int(position_elem.text)
                    value = float(value_elem.text)
                except Exception:
                    continue
                rows.append({
                    "timestamp": start + (position - 1) * resolution,
                    "value": value,
                    "psr_type": psr.text if psr is not None else None,
                    "business_type": business.text if business is not None else None,
                    "resolution_minutes": minutes,
                })
    logger.info(f"Parsed {len(rows)} data rows from XML")
    if not rows:
        raise ValueError("No data rows found – check parameters or API availability.")
    df = pd.DataFrame(rows)
    logger.info(f"Created DataFrame with shape: {df.shape}")
    return df

def fetch_intraday_prices(date: str, zone_name: str) -> pd.DataFrame:
    """
    Intraday prices (`documentType=A44`, `processType=A07`) for one zone.
    Returns a DataFrame with the target date only, localised to the country's timezone.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    # For intraday prices, request the full target day (00:00 to 23:00)
    period_start = target_date.strftime("%Y%m%d0000")
    period_end = target_date.strftime("%Y%m%d2300")

    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    params = {
        "documentType": "A44",
        "processType": "A07",  # Intraday process
        "in_Domain": eic_code,
        "out_Domain": eic_code,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    
    # Log the exact parameters being sent
    logger.info(f"ENTSO-E API parameters (intraday): {params}")
    logger.info(f"Requesting intraday data for {backend_zone_name} (EIC: {eic_code})")
    logger.info(f"Time period: {period_start} to {period_end}")
    
    # Check if we're requesting a future date
    current_date = datetime.now().date()
    if target_date.date() > current_date:
        logger.warning(f"Requesting intraday data for future date {date} (current date: {current_date})")
        logger.warning("ENTSO-E typically only provides historical data with 1-2 day delay")

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            logger.info(f"Attempt {attempt + 1}: Making ENTSO-E API request for {backend_zone_name} (intraday)")
            xml = _query_entsoe(params)
            logger.info(f"Successfully got XML response, length: {len(xml)}")
            df = _xml_timeseries_to_df(xml, value_tag="price.amount")
            logger.info(f"Successfully parsed XML to DataFrame, shape: {df.shape}")
            
            # Check if we got an empty DataFrame (acknowledgement document)
            if df.empty:
                logger.warning(f"No intraday data available for {backend_zone_name} on {date} - received acknowledgement document")
                return df
            
            # Keep only the target date & shift to country's timezone
            timezone_offset = get_timezone_offset_hours(backend_zone_name, target_date)
            df["datetime"] = df["timestamp"] + timedelta(hours=timezone_offset)
            df["price_eur_mwh"] = df["value"]
            df["zone"] = backend_zone_name  # Store backend name for database consistency
            
            # Extract resolution information from the first record
            if not df.empty and "resolution_minutes" in df.columns:
                # Convert resolution from timedelta to minutes
                first_resolution = df.iloc[0]["resolution_minutes"]
                if isinstance(first_resolution, pd.Timedelta):
                    resolution_minutes = int(first_resolution.total_seconds() / 60)
                else:
                    resolution_minutes = first_resolution
                logger.info(f"Intraday data resolution for {backend_zone_name}: {resolution_minutes} minutes")
            else:
                # Default to hourly if no resolution info
                resolution_minutes = 60
                logger.info(f"No resolution info found, defaulting to {resolution_minutes} minutes for {backend_zone_name}")
            
            df["resolution_minutes"] = resolution_minutes
            df = df[["datetime", "zone", "price_eur_mwh", "resolution_minutes"]]
            df = df[df["datetime"].dt.date == target_date.date()]
            logger.info(f"Applied timezone offset of +{timezone_offset} hours for {backend_zone_name}")
            logger.info(f"Final DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
            if not df.empty:
                logger.info(f"Sample data: {df.head().to_dict('records')}")
            else:
                logger.warning(f"No intraday data found for {backend_zone_name} on {date} after filtering")
            return df.reset_index(drop=True)
        except requests.exceptions.RequestException as err:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # exponential back-off
                continue
            raise RuntimeError(f"Failed after {max_attempts} attempts: {err}")

def fetch_day_ahead_prices(date: str, zone_name: str) -> pd.DataFrame:
    """
    Day-ahead prices (`documentType=A44`, `processType=A01`) for one zone.
    Returns a DataFrame with the target date only, localised to the country's timezone.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    # For day-ahead prices, request the full target day (00:00 to 23:00)
    # This ensures we get all 24 hours of day-ahead prices
    period_start = target_date.strftime("%Y%m%d0000")
    period_end = target_date.strftime("%Y%m%d2300")

    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    params = {
        "documentType": "A44",
        "processType": "A01",  # Day-ahead auction process
        "in_Domain": eic_code,
        "out_Domain": eic_code,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    
    # Log the exact parameters being sent
    logger.info(f"ENTSO-E API parameters: {params}")
    logger.info(f"Requesting data for {backend_zone_name} (EIC: {eic_code})")
    logger.info(f"Time period: {period_start} to {period_end}")
    
    # Check if we're requesting a future date
    current_date = datetime.now().date()
    if target_date.date() > current_date:
        logger.warning(f"Requesting data for future date {date} (current date: {current_date})")
        logger.warning("ENTSO-E typically only provides historical data with 1-2 day delay")

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            logger.info(f"Attempt {attempt + 1}: Making ENTSO-E API request for {backend_zone_name}")
            xml = _query_entsoe(params)
            logger.info(f"Successfully got XML response, length: {len(xml)}")
            df = _xml_timeseries_to_df(xml, value_tag="price.amount")
            logger.info(f"Successfully parsed XML to DataFrame, shape: {df.shape}")
            
            # Check if we got an empty DataFrame (acknowledgement document)
            if df.empty:
                logger.warning(f"No data available for {backend_zone_name} on {date} - received acknowledgement document")
                return df
            
            # Keep only the target date & shift to country's timezone
            timezone_offset = get_timezone_offset_hours(backend_zone_name, target_date)
            df["datetime"] = df["timestamp"] + timedelta(hours=timezone_offset)
            df["price_eur_mwh"] = df["value"]
            df["zone"] = backend_zone_name  # Store backend name for database consistency
            
            # Extract resolution information from the first record
            if not df.empty and "resolution_minutes" in df.columns:
                # Convert resolution from timedelta to minutes
                first_resolution = df.iloc[0]["resolution_minutes"]
                if isinstance(first_resolution, pd.Timedelta):
                    resolution_minutes = int(first_resolution.total_seconds() / 60)
                else:
                    resolution_minutes = first_resolution
                logger.info(f"Data resolution for {backend_zone_name}: {resolution_minutes} minutes")
            else:
                # Default to hourly if no resolution info
                resolution_minutes = 60
                logger.info(f"No resolution info found, defaulting to {resolution_minutes} minutes for {backend_zone_name}")
            
            df["resolution_minutes"] = resolution_minutes
            df = df[["datetime", "zone", "price_eur_mwh", "resolution_minutes"]]
            df = df[df["datetime"].dt.date == target_date.date()]
            logger.info(f"Applied timezone offset of +{timezone_offset} hours for {backend_zone_name}")
            logger.info(f"Final DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
            if not df.empty:
                logger.info(f"Sample data: {df.head().to_dict('records')}")
            else:
                logger.warning(f"No data found for {backend_zone_name} on {date} after filtering")
            return df.reset_index(drop=True)
        except requests.exceptions.RequestException as err:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # exponential back-off
                continue
            raise RuntimeError(f"Failed after {max_attempts} attempts: {err}")

def fetch_total_load(date: str, zone_name: str) -> pd.DataFrame:
    """
    Actual Total Load – `documentType=A65`, `processType=A16` (Realised). 6.1.A
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    xml = _query_entsoe({
        "documentType": "A65", 
        "processType": "A16",
        "outBiddingZone_Domain": eic_code,
        "periodStart": start_time.strftime("%Y%m%d%H%M"),
        "periodEnd": end_time.strftime("%Y%m%d%H%M"),
    })
    df = _xml_timeseries_to_df(xml)
    
    # Check if DataFrame is empty (no data available)
    if df.empty:
        # Return empty DataFrame with correct columns for consistency
        return pd.DataFrame(columns=["timestamp", "zone", "load_mwh"])
    
    # Rename value column and add metadata
    df = df.rename(columns={"value": "load_mwh"})
    df["zone"] = backend_zone_name  # Store backend name for database consistency
    return df[["timestamp", "zone", "load_mwh"]]

def fetch_generation_mix(date: str, zone_name: str, psr_type: Optional[str] = None) -> pd.DataFrame:
    """
    Actual Generation per Production Type – `A75` (16.1.B&C).
    Optional psr_type filter (e.g. 'B02' = Solar, see code-list PDF).
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    params = {
        "documentType": "A75", 
        "processType": "A16",
        "in_Domain": eic_code,
        "periodStart": start_time.strftime("%Y%m%d%H%M"),
        "periodEnd": end_time.strftime("%Y%m%d%H%M"),
    }
    if psr_type:
        params["psrType"] = psr_type
    xml = _query_entsoe(params)
    df = _xml_timeseries_to_df(xml)
    
    # Check if DataFrame is empty (no data available)
    if df.empty:
        # Return empty DataFrame with correct columns for consistency
        return pd.DataFrame(columns=["timestamp", "zone", "generation_mw", "psr_type", "resolution_minutes"])
    
    # Rename value column and add metadata
    # NOTE: ENTSOE returns instantaneous power (MW), not energy (MWh)
    df = df.rename(columns={"value": "generation_mw"})
    df["zone"] = backend_zone_name  # Store backend name for database consistency
    df["psr_type"] = psr_type  # Add the PSR type to the DataFrame
    return df[["timestamp", "zone", "generation_mw", "psr_type", "resolution_minutes"]]


def fetch_all_generation_types(date: str, zone_name: str) -> pd.DataFrame:
    """
    Fetch all available generation types for a given country and date.
    Returns a DataFrame with all generation sources for each timestamp.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    # Common generation types to fetch (including wind types B18 and B19)
    common_psr_types = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10",
        "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20"
    ]
    
    all_data = []
    
    for psr_type in common_psr_types:
        try:
            params = {
                "documentType": "A75", 
                "processType": "A16",
                "in_Domain": eic_code,
                "periodStart": start_time.strftime("%Y%m%d%H%M"),
                "periodEnd": end_time.strftime("%Y%m%d%H%M"),
                "psrType": psr_type
            }
            xml = _query_entsoe(params)
            df = _xml_timeseries_to_df(xml)
            
            if not df.empty:
                # Rename value column and add metadata
                # NOTE: ENTSOE returns instantaneous power (MW), not energy (MWh)
                df = df.rename(columns={"value": "generation_mw"})
                df["zone"] = backend_zone_name
                df["psr_type"] = psr_type
                all_data.append(df)
                
        except Exception as e:
            logger.warning(f"Failed to fetch generation data for PSR type {psr_type} for {zone_name}: {str(e)}")
            continue
    
    if not all_data:
        # Return empty DataFrame with correct columns for consistency
        return pd.DataFrame(columns=["timestamp", "zone", "generation_mw", "psr_type", "resolution_minutes"])
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df[["timestamp", "zone", "generation_mw", "psr_type", "resolution_minutes"]]

def fetch_activated_balancing(date: str, zone_name: str, business_type: str = "A96") -> pd.DataFrame:
    """
    Activated Balancing Energy – `A83` (17.1.E).
    business_type: A96 = FRR, A97 = mFRR, A98 = RR.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    xml = _query_entsoe({
        "documentType": "A83",
        "businessType": business_type,
        "controlArea_Domain": eic_code,
        "periodStart": start_time.strftime("%Y%m%d%H%M"),
        "periodEnd": end_time.strftime("%Y%m%d%H%M"),
    })
    df = _xml_timeseries_to_df(xml)
    
    # Check if DataFrame is empty (no data available)
    if df.empty:
        # Return empty DataFrame with correct columns for consistency
        return pd.DataFrame(columns=["timestamp", "zone", "activated_mwh", "business_type"])
    
    # Rename value column and add metadata
    df = df.rename(columns={"value": "activated_mwh"})
    df["zone"] = backend_zone_name  # Store backend name for database consistency
    df["business_type"] = business_type  # Add the business type to the DataFrame
    return df[["timestamp", "zone", "activated_mwh", "business_type"]]

def _check_data_availability(date: str, document_type: str) -> None:
    """
    Check if data might be available for the given date and document type.
    ENTSO-E data typically has a delay of 1-2 days.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    today = datetime.now().date()
    
    # Check if date is in the future
    if target_date.date() > today:
        raise ValueError(f"Date {date} is in the future. ENTSO-E data is not available for future dates.")
    
    # Check if date is too recent (within last 2 days)
    if (today - target_date.date()).days <= 2:
        logger.warning(f"Date {date} is very recent. ENTSO-E data might not be available yet (typically 1-2 day delay).")
    
    # Document type specific checks - be more lenient for imbalance data
    if document_type in ["A85", "A86"]:  # Imbalance data
        if (today - target_date.date()).days <= 1:
            logger.warning(f"Imbalance data (document type {document_type}) for {date} might not be available yet. ENTSO-E typically publishes this data with a 1-2 day delay.")
        # Don't raise an error for recent dates, just log a warning

def fetch_imbalance_volume(date: str, zone_name: str) -> Optional[pd.DataFrame]:
    """
    Imbalance Volumes – `A86`.
    Returns None if data is not available for the specified country.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    # Check data availability
    _check_data_availability(date, "A86")

    try:
        xml = _query_entsoe({
            "documentType": "A86",
            "controlArea_Domain": eic_code,
            "periodStart": start_time.strftime("%Y%m%d%H%M"),
            "periodEnd": end_time.strftime("%Y%m%d%H%M"),
        })
        df = _xml_timeseries_to_df(xml)
        
        # Check if DataFrame is empty (no data available)
        if df.empty:
            logger.warning(f"No imbalance volume data available for {backend_zone_name} on {date}")
            return None
        
        # Rename value column and add metadata
        df = df.rename(columns={"value": "imbalance_MWh"})
        df["zone"] = backend_zone_name  # Store backend name for database consistency
        
        # Apply timezone offset based on country
        timezone_offset = get_timezone_offset_hours(backend_zone_name, target_date)
        df["timestamp"] = df["timestamp"] + timedelta(hours=timezone_offset)
        
        logger.info(f"Applied timezone offset of +{timezone_offset} hours for {backend_zone_name}")
        return df[["timestamp", "zone", "imbalance_MWh"]]
    except ValueError as e:
        # Check if this is a "no data" error for this specific country
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty']):
            logger.warning(f"Imbalance volume data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            # Re-raise other ValueError exceptions
            raise
    except Exception as e:
        # For other exceptions, check if it's related to data availability
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty', 'invalid']):
            logger.warning(f"Imbalance volume data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            raise ValueError(f"Failed to fetch imbalance volume for {backend_zone_name} on {date}: {str(e)}")

def fetch_imbalance_price(date: str, zone_name: str) -> Optional[pd.DataFrame]:
    """
    Imbalance Prices – `A85`.
    Returns None if data is not available for the specified country.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    # Check data availability
    _check_data_availability(date, "A85")

    try:
        xml = _query_entsoe({
            "documentType": "A85",
            "controlArea_Domain": eic_code,
            "periodStart": start_time.strftime("%Y%m%d%H%M"),
            "periodEnd": end_time.strftime("%Y%m%d%H%M"),
        })
        
        # Try different value tags for price data
        value_tags_to_try = ["imbalance_Price.amount", "price.amount", "amount", "price", "quantity"]
        
        for value_tag in value_tags_to_try:
            try:
                logger.info(f"Trying to parse price data with value tag: {value_tag}")
                df = _xml_timeseries_to_df(xml, value_tag=value_tag)
                df = df.rename(columns={"value": "imbalance_EUR_per_MWh"})
                df["zone"] = backend_zone_name  # Store backend name for database consistency
                
                # Apply timezone offset based on country
                timezone_offset = get_timezone_offset_hours(backend_zone_name, target_date)
                df["timestamp"] = df["timestamp"] + timedelta(hours=timezone_offset)
                
                logger.info(f"Successfully parsed price data with tag '{value_tag}', found {len(df)} rows")
                logger.info(f"Applied timezone offset of +{timezone_offset} hours for {backend_zone_name}")
                return df[["timestamp", "zone", "imbalance_EUR_per_MWh"]]
            except Exception as e:
                logger.warning(f"Failed to parse with tag '{value_tag}': {str(e)}")
                continue
        
        # If all tags fail, check if it's a data availability issue
        logger.warning(f"Could not parse price data for {backend_zone_name} on {date} with any of the tried tags: {value_tags_to_try}")
        return None
        
    except ValueError as e:
        # Check if this is a "no data" error for this specific country
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty']):
            logger.warning(f"Imbalance price data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            # Re-raise other ValueError exceptions
            raise
    except Exception as e:
        # For other exceptions, check if it's related to data availability
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty', 'invalid']):
            logger.warning(f"Imbalance price data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            raise ValueError(f"Failed to fetch imbalance price for {backend_zone_name} on {date}: {str(e)}")

def fetch_day_ahead_prices_for_specific_zone(date: str, backend_zone_name: str) -> pd.DataFrame:
    """
    Fetch day-ahead prices for a specific backend zone name (e.g., 'Sweden 3', 'Denmark 2').
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    period_start = target_date.strftime("%Y%m%d0000")
    period_end = target_date.strftime("%Y%m%d2300")

    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    params = {
        "documentType": "A44",
        "processType": "A01",  # Day-ahead auction process
        "in_Domain": eic_code,
        "out_Domain": eic_code,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    
    logger.info(f"Fetching day-ahead prices for specific zone: {backend_zone_name} (EIC: {eic_code})")
    
    # Check if we're requesting a future date
    current_date = datetime.now().date()
    if target_date.date() > current_date:
        logger.warning(f"Requesting data for future date {date} (current date: {current_date})")
        return pd.DataFrame(columns=["datetime", "zone", "price_eur_mwh"])  # Return empty for future dates

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            logger.info(f"Attempt {attempt + 1}: Making ENTSO-E API request for {backend_zone_name}")
            xml = _query_entsoe(params)
            df = _xml_timeseries_to_df(xml, value_tag="price.amount")
            
            if df.empty:
                logger.warning(f"No data available for {backend_zone_name} on {date}")
                return df
            
            # Process the data
            timezone_offset = get_timezone_offset_hours(backend_zone_name, target_date)
            df["datetime"] = df["timestamp"] + timedelta(hours=timezone_offset)
            df["price_eur_mwh"] = df["value"]
            df["zone"] = backend_zone_name
            df = df[["datetime", "zone", "price_eur_mwh"]]
            df = df[df["datetime"].dt.date == target_date.date()]
            
            logger.info(f"Successfully fetched {len(df)} records for {backend_zone_name}")
            return df.reset_index(drop=True)
            
        except requests.exceptions.RequestException as err:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
                continue
            logger.error(f"Failed to fetch data for {backend_zone_name}: {err}")
            return pd.DataFrame(columns=["datetime", "zone", "price_eur_mwh"])
    
    return pd.DataFrame(columns=["datetime", "zone", "price_eur_mwh"])

def fetch_load_forecast_day_ahead(date: str, zone_name: str) -> Optional[pd.DataFrame]:
    """
    Load Forecast Day Ahead – `A65`.
    Returns None if data is not available for the specified country.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    # Check data availability
    _check_data_availability(date, "A65")

    try:
        xml = _query_entsoe({
            "documentType": "A65",
            "processType": "A01",  # Day ahead
            "outBiddingZone_Domain": eic_code,
            "periodStart": start_time.strftime("%Y%m%d%H%M"),
            "periodEnd": end_time.strftime("%Y%m%d%H%M"),
        })
        df = _xml_timeseries_to_df(xml)
        
        # Check if DataFrame is empty (no data available)
        if df.empty:
            logger.warning(f"No load forecast data available for {backend_zone_name} on {date}")
            return None
        
        # Rename value column and add metadata
        df = df.rename(columns={"value": "load_forecast_MW"})
        df["zone"] = backend_zone_name
        df["data_type"] = "load_forecast"
        
        # Apply timezone offset based on country
        timezone_offset = get_timezone_offset_hours(backend_zone_name, target_date)
        df["timestamp"] = df["timestamp"] + timedelta(hours=timezone_offset)
        
        logger.info(f"Successfully fetched load forecast for {backend_zone_name} on {date}")
        return df[["timestamp", "zone", "load_forecast_MW", "data_type"]]
        
    except ValueError as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty']):
            logger.warning(f"Load forecast data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            raise
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty', 'invalid']):
            logger.warning(f"Load forecast data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            raise ValueError(f"Failed to fetch load forecast for {backend_zone_name} on {date}: {str(e)}")

def fetch_installed_capacity_per_type(date: str, zone_name: str, psr_type: Optional[str] = None) -> pd.DataFrame:
    """
    Installed Capacity Per Production Type – `A68` with `A33` (14.1.A).
    Returns installed generation capacity by production type.
    
    Note: Changed from A71 to A68 as A71 is for generation forecast, not installed capacity.
    A68 is the correct document type for installed generation capacity per type.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    params = {
        "documentType": "A68",  # Installed generation capacity per type (was A71)
        "processType": "A33",   # Year ahead
        "in_Domain": eic_code,
        "periodStart": start_time.strftime("%Y%m%d%H%M"),
        "periodEnd": end_time.strftime("%Y%m%d%H%M"),
    }
    
    if psr_type:
        params["psrType"] = psr_type
    
    try:
        xml = _query_entsoe(params)
        df = _xml_timeseries_to_df(xml)
        
        if df.empty:
            logger.warning(f"No installed capacity data available for {backend_zone_name} on {date}")
            return pd.DataFrame(columns=["timestamp", "zone", "installed_capacity_mw", "psr_type"])
        
        # Rename value column - this is installed capacity in MW
        df = df.rename(columns={"value": "installed_capacity_mw"})
        df["zone"] = backend_zone_name
        
        # For A68 documents, the PSR type might not be extracted from XML correctly
        # Since we passed it as a parameter, explicitly set it
        if psr_type:
            df["psr_type"] = psr_type
        
        return df[["timestamp", "zone", "installed_capacity_mw", "psr_type"]]
    
    except Exception as e:
        logger.warning(f"Failed to fetch installed capacity for {backend_zone_name}: {str(e)}")
        return pd.DataFrame(columns=["timestamp", "zone", "installed_capacity_mw", "psr_type"])


def fetch_installed_capacity_all_types(date: str, zone_name: str) -> pd.DataFrame:
    """
    Fetch installed capacity for all production types.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    # Common generation types
    common_psr_types = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10",
        "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20"
    ]
    
    all_data = []
    
    for psr_type in common_psr_types:
        try:
            df = fetch_installed_capacity_per_type(date, zone_name, psr_type)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            logger.debug(f"No installed capacity data for PSR type {psr_type}: {str(e)}")
            continue
    
    if not all_data:
        return pd.DataFrame(columns=["timestamp", "zone", "installed_capacity_mw", "psr_type"])
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def fetch_generation_forecast_day_ahead(date: str, zone_name: str) -> Optional[pd.DataFrame]:
    """
    Generation Forecast Day Ahead – `A71`.
    Returns None if data is not available for the specified country.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    # Check data availability
    _check_data_availability(date, "A71")

    try:
        xml = _query_entsoe({
            "documentType": "A71",
            "processType": "A01",  # Day ahead
            "in_Domain": eic_code,
            "periodStart": start_time.strftime("%Y%m%d%H%M"),
            "periodEnd": end_time.strftime("%Y%m%d%H%M"),
        })
        df = _xml_timeseries_to_df(xml)
        
        # Check if DataFrame is empty (no data available)
        if df.empty:
            logger.warning(f"No generation forecast data available for {backend_zone_name} on {date}")
            return None
        
        # Rename value column and add metadata
        df = df.rename(columns={"value": "generation_forecast_MW"})
        df["zone"] = backend_zone_name
        df["data_type"] = "generation_forecast"
        
        # Apply timezone offset based on country
        timezone_offset = get_timezone_offset_hours(backend_zone_name, target_date)
        df["timestamp"] = df["timestamp"] + timedelta(hours=timezone_offset)
        
        logger.info(f"Successfully fetched generation forecast for {backend_zone_name} on {date}")
        return df[["timestamp", "zone", "generation_forecast_MW", "data_type"]]
        
    except ValueError as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty']):
            logger.warning(f"Generation forecast data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            raise
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty', 'invalid']):
            logger.warning(f"Generation forecast data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            raise ValueError(f"Failed to fetch generation forecast for {backend_zone_name} on {date}: {str(e)}")

def fetch_day_ahead_prices_for_country_all_zones(date: str, country_identifier: str) -> pd.DataFrame:
    """
    Fetch day-ahead prices for ALL zones within a country.
    Args:
        date: Date string in YYYY-MM-DD format
        country_identifier: Country name, ISO code, or zone identifier
    Returns:
        DataFrame with data from all zones in the country
    """
    from app.utils.zone_mapper import get_all_zones_for_country, normalize_zone
    
    try:
        # Get all zones for this country
        iso_code = normalize_zone(country_identifier)
        all_zones = get_all_zones_for_country(iso_code)
        logger.info(f"Fetching day-ahead prices for all zones in {country_identifier}: {all_zones}")
        
        all_dataframes = []
        
        # Fetch data for each zone separately
        for zone in all_zones:
            try:
                logger.info(f"Fetching data for zone: {zone}")
                df = fetch_day_ahead_prices_for_specific_zone(date, zone)
                if not df.empty:
                    all_dataframes.append(df)
                    logger.info(f"Successfully fetched {len(df)} records for {zone}")
                else:
                    logger.warning(f"No data available for {zone} on {date}")
            except Exception as e:
                logger.error(f"Failed to fetch data for zone {zone}: {str(e)}")
                continue
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            logger.info(f"Combined data from {len(all_dataframes)} zones, total records: {len(combined_df)}")
            return combined_df
        else:
            logger.warning(f"No data available for any zone in {country_identifier} on {date}")
            return pd.DataFrame(columns=["datetime", "zone", "price_eur_mwh"])
            
    except Exception as e:
        logger.error(f"Error fetching data for country {country_identifier}: {str(e)}")
        return pd.DataFrame(columns=["datetime", "zone", "price_eur_mwh"])

def fetch_day_ahead_for_all_zones(date_str: str) -> pd.DataFrame:
    """
    Fetch day-ahead prices for all configured bidding zones.
    """
    results = []
    for backend_zone_name in BIDDING_ZONES:
        try:
            # Convert backend zone name to frontend name for the fetch function
            frontend_zone_name = get_frontend_country_name(backend_zone_name)
            df = fetch_day_ahead_prices(date_str, frontend_zone_name)
            
            # Only add non-empty DataFrames
            if not df.empty:
                results.append(df)
                logger.info(f"Successfully fetched data for {backend_zone_name}: {len(df)} records")
            else:
                logger.warning(f"No data available for {backend_zone_name} on {date_str}")
        except Exception as e:
            logger.warning(f"Failed to fetch prices for {backend_zone_name}: {e}")
            continue

    if not results:
        logger.warning(f"No data available for any zone on {date_str}")
        # Return empty DataFrame instead of raising error
        return pd.DataFrame(columns=["timestamp", "zone", "price_eur_mwh"])

    logger.info(f"Successfully fetched data for {len(results)} zones")
    return pd.concat(results, ignore_index=True)

def fetch_cross_border_physical_flows(date: str, from_zone: str, to_zone: str) -> Optional[pd.DataFrame]:
    """
    Cross-Border Physical Flows – `A11`.
    Returns physical flows between two bidding zones.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country names to backend country names
    backend_from_zone = get_backend_country_name(from_zone)
    backend_to_zone = get_backend_country_name(to_zone)
    
    from_eic_code = BIDDING_ZONES.get(backend_from_zone)
    to_eic_code = BIDDING_ZONES.get(backend_to_zone)
    
    if not from_eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_from_zone}")
    if not to_eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_to_zone}")

    # Check data availability
    _check_data_availability(date, "A11")

    try:
        xml = _query_entsoe({
            "documentType": "A11",
            "in_Domain": from_eic_code,
            "out_Domain": to_eic_code,
            "periodStart": start_time.strftime("%Y%m%d%H%M"),
            "periodEnd": end_time.strftime("%Y%m%d%H%M"),
        })
        
        df = _xml_timeseries_to_df(xml)
        
        # Check if DataFrame is empty (no data available)
        if df.empty:
            logger.warning(f"No cross-border physical flow data available from {backend_from_zone} to {backend_to_zone} on {date}")
            return None
        
        # Rename value column and add metadata
        df = df.rename(columns={"value": "flow_MW"})
        df["from_zone"] = backend_from_zone
        df["to_zone"] = backend_to_zone
        
        # Apply timezone offset based on from_zone
        timezone_offset = get_timezone_offset_hours(backend_from_zone, target_date)
        df["timestamp"] = df["timestamp"] + timedelta(hours=timezone_offset)
        
        logger.info(f"Applied timezone offset of +{timezone_offset} hours for {backend_from_zone}")
        return df[["timestamp", "from_zone", "to_zone", "flow_MW"]]
    except ValueError as e:
        # Check if this is a "no data" error for this specific border
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty']):
            logger.warning(f"Cross-border physical flow data not available from {backend_from_zone} to {backend_to_zone} on {date}: {str(e)}")
            return None
        else:
            # Re-raise other ValueError exceptions
            raise
    except Exception as e:
        # For other exceptions, check if it's related to data availability
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty', 'invalid']):
            logger.warning(f"Cross-border physical flow data not available from {backend_from_zone} to {backend_to_zone} on {date}: {str(e)}")
            return None
        else:
            raise ValueError(f"Failed to fetch cross-border physical flows from {backend_from_zone} to {backend_to_zone} on {date}: {str(e)}")

def fetch_balancing_energy_bids(date: str, zone_name: str, business_type: str = "A81") -> Optional[pd.DataFrame]:
    """
    Balancing Energy Bids – `A81` or `A82`.
    A81: Balancing energy bids (17.1.C)
    A82: Balancing energy bids (alternative format)
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    # Check data availability
    _check_data_availability(date, business_type)

    try:
        xml = _query_entsoe({
            "documentType": business_type,
            "controlArea_Domain": eic_code,
            "periodStart": start_time.strftime("%Y%m%d%H%M"),
            "periodEnd": end_time.strftime("%Y%m%d%H%M"),
        })
        
        df = _xml_timeseries_to_df(xml)
        
        # Check if DataFrame is empty (no data available)
        if df.empty:
            logger.warning(f"No balancing energy bid data available for {backend_zone_name} on {date}")
            return None
        
        # Rename value column and add metadata
        df = df.rename(columns={"value": "bid_volume_MW"})
        df["zone"] = backend_zone_name
        df["business_type"] = business_type
        
        # Apply timezone offset based on country
        timezone_offset = get_timezone_offset_hours(backend_zone_name, target_date)
        df["timestamp"] = df["timestamp"] + timedelta(hours=timezone_offset)
        
        logger.info(f"Applied timezone offset of +{timezone_offset} hours for {backend_zone_name}")
        return df[["timestamp", "zone", "bid_volume_MW", "business_type", "psr_type"]]
    except ValueError as e:
        # Check if this is a "no data" error for this specific country
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty']):
            logger.warning(f"Balancing energy bid data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            # Re-raise other ValueError exceptions
            raise
    except Exception as e:
        # For other exceptions, check if it's related to data availability
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty', 'invalid']):
            logger.warning(f"Balancing energy bid data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            raise ValueError(f"Failed to fetch balancing energy bids for {backend_zone_name} on {date}: {str(e)}")

def fetch_balancing_energy_prices(date: str, zone_name: str) -> Optional[pd.DataFrame]:
    """
    Prices paid by the TSO for activated balancing energy per balancing time unit and per type of reserve – `A84`.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country name to backend country name
    backend_zone_name = get_backend_country_name(zone_name)
    eic_code = BIDDING_ZONES.get(backend_zone_name)
    if not eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_zone_name}")

    # Check data availability
    _check_data_availability(date, "A84")

    try:
        xml = _query_entsoe({
            "documentType": "A84",
            "controlArea_Domain": eic_code,
            "periodStart": start_time.strftime("%Y%m%d%H%M"),
            "periodEnd": end_time.strftime("%Y%m%d%H%M"),
        })
        
        # For price data, we need to use a different value tag
        df = _xml_timeseries_to_df(xml, value_tag="price.amount")
        
        # Check if DataFrame is empty (no data available)
        if df.empty:
            logger.warning(f"No balancing energy price data available for {backend_zone_name} on {date}")
            return None
        
        # Rename value column and add metadata
        df = df.rename(columns={"value": "price_eur_MWh"})
        df["zone"] = backend_zone_name
        
        # Apply timezone offset based on country
        timezone_offset = get_timezone_offset_hours(backend_zone_name, target_date)
        df["timestamp"] = df["timestamp"] + timedelta(hours=timezone_offset)
        
        logger.info(f"Applied timezone offset of +{timezone_offset} hours for {backend_zone_name}")
        return df[["timestamp", "zone", "price_eur_MWh", "psr_type", "business_type"]]
    except ValueError as e:
        # Check if this is a "no data" error for this specific country
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty']):
            logger.warning(f"Balancing energy price data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            # Re-raise other ValueError exceptions
            raise
    except Exception as e:
        # For other exceptions, check if it's related to data availability
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty', 'invalid']):
            logger.warning(f"Balancing energy price data not available for {backend_zone_name} on {date}: {str(e)}")
            return None
        else:
            raise ValueError(f"Failed to fetch balancing energy prices for {backend_zone_name} on {date}: {str(e)}")

def fetch_netted_exchanged_volumes_per_border(date: str, from_zone: str, to_zone: str) -> Optional[pd.DataFrame]:
    """
    Netted and Exchanged Volumes per Border – `A09`.
    Returns netted exchange volumes between two bidding zones.
    """
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_time = target_date.replace(hour=0, minute=0)
    end_time = start_time + timedelta(days=1)
    
    # Convert frontend country names to backend country names
    backend_from_zone = get_backend_country_name(from_zone)
    backend_to_zone = get_backend_country_name(to_zone)
    
    from_eic_code = BIDDING_ZONES.get(backend_from_zone)
    to_eic_code = BIDDING_ZONES.get(backend_to_zone)
    
    if not from_eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_from_zone}")
    if not to_eic_code:
        raise ValueError(f"Unknown bidding zone: {backend_to_zone}")

    # Check data availability
    _check_data_availability(date, "A09")

    try:
        xml = _query_entsoe({
            "documentType": "A09",
            "in_Domain": from_eic_code,
            "out_Domain": to_eic_code,
            "periodStart": start_time.strftime("%Y%m%d%H%M"),
            "periodEnd": end_time.strftime("%Y%m%d%H%M"),
        })
        
        df = _xml_timeseries_to_df(xml)
        
        # Check if DataFrame is empty (no data available)
        if df.empty:
            logger.warning(f"No netted exchange volume data available from {backend_from_zone} to {backend_to_zone} on {date}")
            return None
        
        # Rename value column and add metadata
        df = df.rename(columns={"value": "exchange_volume_MWh"})
        df["from_zone"] = backend_from_zone
        df["to_zone"] = backend_to_zone
        
        # Apply timezone offset based on from_zone
        timezone_offset = get_timezone_offset_hours(backend_from_zone, target_date)
        df["timestamp"] = df["timestamp"] + timedelta(hours=timezone_offset)
        
        logger.info(f"Applied timezone offset of +{timezone_offset} hours for {backend_from_zone}")
        return df[["timestamp", "from_zone", "to_zone", "exchange_volume_MWh"]]
    except ValueError as e:
        # Check if this is a "no data" error for this specific border
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty']):
            logger.warning(f"Netted exchange volume data not available from {backend_from_zone} to {backend_to_zone} on {date}: {str(e)}")
            return None
        else:
            # Re-raise other ValueError exceptions
            raise
    except Exception as e:
        # For other exceptions, check if it's related to data availability
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['no data', 'not available', 'not found', 'empty', 'invalid']):
            logger.warning(f"Netted exchange volume data not available from {backend_from_zone} to {backend_to_zone} on {date}: {str(e)}")
            return None
        else:
            raise ValueError(f"Failed to fetch netted exchange volumes from {backend_from_zone} to {backend_to_zone} on {date}: {str(e)}")
