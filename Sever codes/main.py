from fastapi import FastAPI,Depends,HTTPException, status,Request
from pydantic import BaseModel
import json,numbers
import OCRPredict_TensorFlow_Yolo as ocrPredict
from torchvision import transforms
import os,re,csv,datetime,secrets,logging,threading
import requests
from typing import List
from typing_extensions import Annotated
from sqlalchemy import create_engine,text,bindparam,update,Table, Column, Integer, String, MetaData,DateTime,insert
import configparser
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from uuid import UUID, uuid4
from decimal import Decimal, InvalidOperation
from urllib.parse import quote_plus
import base64
import boto3


class ImageFiles(BaseModel):
    description: str | None = None
    listOfFiles: List[str]

class JobImage(BaseModel):      
    description: str | None = None
    imageType:str
    isImageIdOrJobId:str | None = None
    batchSize:int | None = None
    listOfIds: List[str] | None = None
    additionalFilter:str | None = None
    orderBy:str | None = None

class ImageName(BaseModel):
    description: str | None = None
    imageType:str
    batchSize:int
    listOfNames: List[str]
    
class ECM(BaseModel):
    description: str | None = None
    imageType: str | None = None
    batchSize:int | None = None
    jobStatus:str | None = None
    scheme: str
    server: str
    contextPath:str
    clientId:str
    clientSecret:str
    oAuthUser:str
    oAuthPass:str
    listOfIds: List[str] | None = None
    additionalFilter:str | None = None
    
# path = os.getenv('wflowhome')
# pythonpath = os.path.join(path,'.python')
cfg = configparser.ConfigParser()
# cfg.read(os.path.join(pythonpath,'jobs','ocrSettings.ini'))
cfg.read('ocrSettings.ini')
dbSection = cfg.get('database', 'rdbms', raw=False)
dbDriver = cfg.get('database', 'driver', raw=False)
dbHost = cfg.get('database', 'host', raw=False)
dbPort = cfg.get('database', 'port', raw=False)
dbDatabase = cfg.get('database', 'database', raw=False)
dbUName = cfg.get('database', 'uname', raw=False)
dbPassword = cfg.get('database', 'password', raw=False)
dbPassword = quote_plus(dbPassword)
S3accesskey = cfg.get('S3','aws_access_key_id',raw=False)
S3secretkey = cfg.get('S3','aws_secret_access_key',raw=False)
S3region = cfg.get('S3','region_name',raw=False)
S3bucket = cfg.get('S3','bucket_name',raw=False)
s3 = boto3.client('s3', 
                  aws_access_key_id=S3accesskey,
                  aws_secret_access_key=S3secretkey,
                  region_name=S3region)
engine = create_engine(
    f"{dbSection}+{dbDriver}://{dbUName}:{dbPassword}@{dbHost}:{dbPort}/{dbDatabase}",
    echo=False,
    pool_size=90,
    max_overflow=10,
    pool_recycle=600,
    pool_timeout=600
)
logger = logging.getLogger('OCR_Inference')
logging.basicConfig(filename='ocr.log', level=logging.INFO)

app = FastAPI()
security = HTTPBasic()

def generate_token(url,CI,CS,UN,PW):
    #CI = "fooClientId"
    #CS = "secret"
    #UN = "Admin"
    #PW = "Admin@123"
    oAuthpart = "/oauth/token"
    payload = f"grant_type=password&client_id={CI}&client_secret={CS}&username={UN}&password={PW}"
    headers = {'accept': "application/json",
               "Content-Type": "application/x-www-form-urlencoded"}
    logger.info('Starting to Generate Token for ocrFromECM at'+str(datetime.datetime.now())+" With parameters "+payload+" off url "+url+oAuthpart)     
    try:
        response = requests.request(
            "POST", url+oAuthpart, data=payload, headers=headers)
    except requests.exceptions.RequestException as ex:
        logger.error('Error Gettting Alfresco Token for ocrFromECM at '+str(datetime.datetime.now())+" With Exception "+str(ex))
        return None
    if response.status_code != 200:
        return None
    TokenResponse = response.json()
    logger.info('Completed Gettting Generate Token for ocrFromECM for at '+str(datetime.datetime.now())+" With parameters "+json.dumps(TokenResponse))     
    auth_token = TokenResponse['access_token']
    return auth_token

def get_current_username(credentials: Annotated[HTTPBasicCredentials, Depends(security)],):

    current_username_bytes = credentials.username.encode("utf8")
    userName = cfg.get('Auth', 'user', raw=False)
    correct_username_bytes = userName.encode("utf8")
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    password = cfg.get('Auth', 'password', raw=False)
    correct_password_bytes = password.encode("utf8")
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/ocr/ping")
def ping(request: Request,username: Annotated[str, Depends(get_current_username)],item: str):
    #worker_id = request.scope["worker"]
    logger.info('Started Ping for '+username+' at '+str(datetime.datetime.now()))
    try:
        with engine.connect() as c:
            logger.info("Connection Success")
    except Exception as e:
        logger.info(f'connection failed error:{e}')
    return {"Hello": "World","Name":item}

## Output
# -1 -- Improper Image - not able to identify Region of Interest
# -2 -- Improper Image - Possibly poor resolution or model not trained
# -3 -- Alfresco unable to get the Image
# -4 -- Auth Token is not generated
    
@app.post("/ocr/ocrFromFile")
def doOcr(username: Annotated[str, Depends(get_current_username)],item: ImageFiles):
    resultJson=[]
    logger.info('Started ocrFromFile for '+username+' at '+str(datetime.datetime.now())+" With parameters "+json.dumps(item.model_dump()))
    listOfImages = []
    for eachFile in item.listOfFiles:
        fileObject=os.path.join(pythonpath,'images',eachFile)
        image = ocrPredict.readImageFromFile(fileObject)
        listOfImages.append(image)
       
    regions,empty_image = ocrPredict.roi(listOfImages)
    print(regions)
    print(empty_image)
    gen_text = ocrPredict.predictBatch(regions,empty_image)
    logger.info('OCR in ocrFromFile for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(gen_text))
    for index,eachGenText in enumerate(gen_text) :
        if eachGenText is None :
            eachResultjson = {"imageId":item.listOfFiles[index],"read":"-1"}
        else:
            eachResultjson = parseReadResponse(eachGenText,item.listOfFiles[index],len(item.listOfFiles))
        logger.info('each ocrFromFile for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(eachResultjson))
        resultJson.append(eachResultjson)
    logger.info('Ended ocrFromFile for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(resultJson))
    return resultJson

@app.post("/ocr/ocrFromjobImage")
async def doOcr(request: Request,username: Annotated[str, Depends(get_current_username)],item: JobImage):
    resultJson=[]
    logger.info('Started ocrFromjobImage for '+username+' at '+str(datetime.datetime.now())+" With parameters "+json.dumps(item.model_dump()))
    idList = None
    if item.listOfIds is not None and len(item.listOfIds) > 0 :
        idList = ["'" + sub for sub in item.listOfIds] 
        idList = [sub + "'" for sub in idList]
    if item.imageType.lower() == 'new':
        imageNameList="('New Meter KWH Reading','New Meter MD KVA Reading','New Meter MD KW Reading','New Meter MD KVA Reading','New Meter Export KWH Reading','New Meter Export KVAH Reading','New Meter Export MD KVA Reading','New Meter Export MD KW Reading','New Meter Export PF')"
    else :
        imageNameList="('Old Meter KWH Reading','Old Meter KVAH Reading','Old Meter MD KVA Reading','Old Meter MD KW Reading','Old Meter Export KWH Reading','Old Meter Export KVAH Reading','Old Meter Export MD KVA Reading','Old Meter Export MD KW Reading','Old Meter Export PF')"
    with engine.connect() as conn:
        columnsToSelect = "id,status,ocr_reading,ocr_rdg_unit"
        fromSegment = "from wfms.mwm_job_images"
        whereClause = "where record_status=1 and image_name in "+imageNameList
        if idList is not None :
            whereClause = whereClause + "and "+item.isImageIdOrJobId+" in ("+','.join(idList)+")"
        if item.additionalFilter is not None :
            whereClause = whereClause + "and "+item.additionalFilter
        query = "select "+columnsToSelect+" "+fromSegment+" "+whereClause
        if item.orderBy is not None :
            query = query+ " Order By "+item.orderBy
        logger.info(query)
        unConditionalListCursor = conn.execute(text(query))
        totalRecords = unConditionalListCursor.rowcount
        if item.batchSize is  None or item.batchSize <= 0 : 
            unConditionalList = unConditionalListCursor.fetchall()
        else :
           unConditionalList = unConditionalListCursor.fetchmany(int(item.batchSize))
        toConductOCR = [eachImage for index,eachImage in enumerate(unConditionalList) if eachImage[1] != 'OCR']
        OCRCompleted = [eachImage for index,eachImage in enumerate(unConditionalList) if eachImage[1] == 'OCR']
        tupleToConductOCR = [tuple(toConductOCR) for row in toConductOCR]
        tupleCompletedOCR = [tuple(OCRCompleted) for row in OCRCompleted]
        logger.info('Completed Gettting Records for ocrFromjobImage for '+username+' at '+str(datetime.datetime.now()))
        if len(toConductOCR) == 0 :
            for eachCompletedImage in OCRCompleted:
                resultJson.append({"imageId":eachCompletedImage[0],"read":eachCompletedImage[2],"unit":eachCompletedImage[3],"totalRecords":totalRecords})
            logger.info('Ended ocrFromjobImage for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(resultJson))
            return resultJson
        logger.info('Completed toConductOCR ocrFromjobImage for '+username+' at '+str(datetime.datetime.now())+" No of Records"+str(len(toConductOCR)))
        if idList is not None : 
            idList.clear()
        else :
            idList = []
        for eachImage in toConductOCR:
           idList.append(eachImage[0]) 
        idList = ["'" + sub for sub in idList] 
        idList = [sub + "'" for sub in idList]
        result = conn.execute(text("select id,image_base64 from wfms.mwm_job_images where (status is null or status='') and record_status=1 and id in ("+','.join(idList)+")"))
        listOfImages = []
        MapOfIdsToImages = []
        for index,eachRecord in enumerate(result.mappings()):
            identity = eachRecord["id"]
            base64String = eachRecord["image_base64"]
            image = ocrPredict.decodeBitmaptoimage(base64String)
            listOfImages.append(image)
            MapOfIdsToImages.append(identity)
    regions,empty_image = ocrPredict.roi(listOfImages)
    gen_text = ocrPredict.predictBatch(regions,empty_image)
    logger.info('OCR in ocrFromjobImage for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(gen_text))
    for index,eachGenText in enumerate(gen_text) :
        if eachGenText is None :
            eachResultjson = {"imageId":MapOfIdsToImages[index],"read":"-1","unit":""}
        else:
            eachResultjson = parseReadResponse(eachGenText,MapOfIdsToImages[index],len(MapOfIdsToImages))
        logger.info('each ocrFromjobImage for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(eachResultjson))
        resultJson.append(eachResultjson)
    logger.info('Before Update ocrFromjobImage for '+username+' at '+str(datetime.datetime.now())+" No of Records"+str(len(toConductOCR))) 
    updateJobImage(resultJson,"py_Mobile")
    for eachCompletedImage in OCRCompleted:
      resultJson.append({"imageId":eachCompletedImage[0],"read":eachCompletedImage[2],"unit":eachCompletedImage[3],"totalRecords":totalRecords})  
    logger.info('Ended ocrFromjobImage for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(resultJson))
    return resultJson

@app.post("/ocr/ocrFromImageName")
def doOcr(username: Annotated[str, Depends(get_current_username)],item: ImageName):
    resultJson=[]
    imageNameList=[]
    logger.info('Started ocrFromImageName for '+username+' at '+str(datetime.datetime.now())+" With parameters "+json.dumps(item.model_dump()))
    if item.imageType.lower() == 'new':
        for index,eachimageName in enumerate(item.listOfNames):
            imageNameList.append("New "+eachimageName.replace("_"," "))
    elif item.imageType.lower() == 'old':
        for index,eachimageName in enumerate(item.listOfNames):
            imageNameList.append("Old "+eachimageName.replace("_"," "))
    else :
        for index,eachimageName in enumerate(item.listOfNames):
            imageNameList.append("Old "+eachimageName.replace("_"," "))
            imageNameList.append("New "+eachimageName.replace("_"," "))
    imageNameList = ["'" + sub for sub in imageNameList] 
    imageNameList = [sub + "'" for sub in imageNameList]
    with engine.connect() as conn:
        curResult = conn.execute(text("select id,image_base64 from wfms.mwm_job_images where (status is null or status='') and record_status=1 and image_name in ("+','.join(imageNameList)+") order by job_id,create_date desc FOR UPDATE SKIP LOCKED"))
        totalRecords = curResult.rowcount
        result = curResult.fetchmany(int(item.batchSize))
        logger.info('Completed Gettting Records for ocrFromImageName for '+username+' at '+str(datetime.datetime.now())+" With parameters "+str(totalRecords)+' With Batch Size : '+str(len(result)))
        listOfImages = []
        MapOfIdsToImages = []
        for eachRecord in result:
            identity = eachRecord[0]
            base64String = eachRecord[1]
            image = ocrPredict.decodeBitmaptoimage(base64String)
            listOfImages.append(image)
            MapOfIdsToImages.append(identity)
    regions,empty_image = ocrPredict.roi(listOfImages)
    gen_text = ocrPredict.predictBatch(regions,empty_image)
    logger.info('OCR in ocrFromImageName for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(gen_text))
    for index,eachGenText in enumerate(gen_text) :
        if eachGenText is None :
            eachResultjson = {"imageId":MapOfIdsToImages[index],"read":"-1"}
        else:
            eachResultjson = parseReadResponse(eachGenText,MapOfIdsToImages[index],len(MapOfIdsToImages))
        logger.info('each ocrFromjobImage for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(eachResultjson))
        resultJson.append(eachResultjson)
    logger.info('Before Update ocrFromImageName for '+username+' at '+str(datetime.datetime.now())+" No of Records"+str(len(resultJson)))
    updateJobImage(resultJson,'py_scheduler')
    logger.info('Ended ocrFromImageName for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(resultJson))
    return resultJson

@app.post("/ocr/ocrFromBase64csvFile")
def doOcr(username: Annotated[str, Depends(get_current_username)],item: ImageFiles):
    mapOfbase64String = readFromcsvmultirow(os.path.join(pythonpath,item.listOfFiles[0]))
    logger.info('Started ocrFromBase64csvFile for '+username+' at '+str(datetime.datetime.now())+" With parameters "+json.dumps(item.model_dump()))
    listOfImages = []
    for eachIndex in mapOfbase64String.keys():
        eachbase64String=mapOfbase64String.get(eachIndex)
        image = ocrPredict.decodeBitmaptoimage(eachbase64String)
        listOfImages.append(image)
    regions,empty_image = ocrPredict.roi(listOfImages)
    gen_text = ocrPredict.predictBatch(regions,empty_image)
    logger.info('OCR in ocrFromBase64csvFile for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(gen_text))
    for index,eachGenText in enumerate(gen_text) :
        if eachGenText is None :
           eachResultjson = {"imageId":str(index),"read":"-1"}
        else :
            eachResultjson = parseReadResponse(eachGenText,str(index),totalRecords)
        resultJson.append(eachResultjson)
    logger.info('Ended ocrFromBase64csvFile for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(resultJson))
    return resultJson

@app.post("/ocr/ocrFromECM")
def doOcr(username: Annotated[str, Depends(get_current_username)],item: ECM):
    logger.info('Started ocrFromECM for '+username+' at '+str(datetime.datetime.now())+" With parameters "+json.dumps(item.model_dump()))
    resultJson=[]
    url=item.scheme+'://'+item.server+'/'+item.contextPath
    if item.imageType is not None :
        if item.batchSize is None :
            item.batchSize = 10
        columnsToSelect = "idrov.jobid,idrov.alfrescoid ,idrov.meterreading ,idrov.alfrescoid_kwh ,idrov.kwh_meterreading ,idrov.alfrescoid_kvah ,idrov.kvah_meterreading ,idrov.alfrescoid_kva ,idrov.kva_meterreading ,idrov.alfrescoid_mdkw ,idrov.mdk_wmeterreading ,idrov.alfrescoid_kvahexp ,idrov.kvahexp_meterreading ,idrov.alfrescoid_pf ,idrov.pf_meterreading ,idrov.alfrescoid_mdkwexp ,idrov.mdkwexp_meterreading ,idrov.alfrescoid_mdkvaexp, idrov.mdkvaexp_meterreading,idrov.alfrescoid_kwhexp,idrov.kwhexp_meterreading "
        if item.imageType.lower() == 'old' :
            fromSegment  = " from wfms.o_image_data_recon_ocr_view idrov "
        elif item.imageType.lower() == 'new':
            fromSegment = " from wfms.n_image_data_recon_ocr_view idrov "
        elif item.imageType.lower() == 'retake':                                                #idrov.alfrescoid_kwh
            columnsToSelect = "idrov.jobid,idrov.alfrescoid ,idrov.meterreading ,'0' as mmrRead,idrov.alfrescoid_kwh as kwh_alfresco,idrov.kwh_readings,idrov.kwh_meterreading,idrov.kvah_alfresco ,idrov.kvah_readings ,idrov.kvah_meterreading,idrov.mdkv_alfresco ,idrov.mdkv_readings ,idrov.kva_meterreading,idrov.mdkw_alfresco ,idrov.mdkw_readings ,idrov.mdk_wmeterreading,idrov.pf_alfresco ,idrov.pf_value ,idrov.pf_meterreading,idrov.export_kwh_alfresco ,idrov.export_kwh_reading ,idrov.kwhexp_meterreading,idrov.export_kvah_alfresco ,idrov.export_kvah_reading ,idrov.kvahexp_meterreading,idrov.export_mdkv_alfresco, idrov.export_mdkv_reading,idrov.mdkvaexp_meterreading,idrov.export_mdkw_alfresco,idrov.export_mdkw_reading,idrov.mdkwexp_meterreading,idrov.export_pf_alfrsco,idrov.export_pf_value,'0' as pfValue"
            fromSegment = " from wfms.o_staging_image_data_recon_ocr_view idrov "
        else :
            return []
        additionalFilter = ""
        if item.additionalFilter is not None and item.additionalFilter:
            additionalFilter  = " and "+item.additionalFilter
        whereClause = " where job_status = '"+item.jobStatus+"'"+additionalFilter
        orderByClause = " order by jobid desc"
        query = "select "+columnsToSelect+" "+fromSegment+" "+whereClause+" "+orderByClause
        logger.info("The Query being Fired is  ::::"+query)
        with engine.connect() as conn:
            curResult = conn.execute(text(query))
            totalRecords = curResult.rowcount
            logger.info("total Records is   ::::"+str(totalRecords))
            if totalRecords == 0 :
                return []
            result = curResult.fetchmany(int(item.batchSize))
            logger.info('Completed Gettting Records for ocrFromECM for '+username+' at '+str(datetime.datetime.now())+" With parameters Total Records in DB :"+str(totalRecords)+' of batch Size :'+str(len(result)))
            printablequeryResults = [tuple(row) for row in result]
            logger.info(' The SQL Output is '+json.dumps(printablequeryResults, default=str ))
            mapOfAlfrescoToMeterRead = []
            for eachRecord in result:
                jobId = eachRecord[0]
                newEachRecord = eachRecord[1:]
                counter = 0;
                columnCounter = 2
                if item.imageType.lower() == 'retake':
                    columnCounter = 3
                for index,eachcolumn in enumerate(newEachRecord) :
                    if index % columnCounter == 0 :
                       alfrescoId =  newEachRecord[index]
                       meterRead = newEachRecord[index+1]
                       mmrMeterRead = None
                       if item.imageType.lower() == 'retake':
                            mmrMeterRead = newEachRecord[index+2]
                       if alfrescoId is not None and alfrescoId != 'Image Not Taken' and alfrescoId != 'NA':
                            eachId = {"alfrescoId":alfrescoId,"meterRead":meterRead,"mmrMeterRead":mmrMeterRead,"jobId":jobId,"readType":str(counter)}
                            mapOfAlfrescoToMeterRead.append(eachId)
                       counter = counter+1
            if item.listOfIds is None:
                item.listOfIds = []
            if len(mapOfAlfrescoToMeterRead) > 0:
                for eachMap in mapOfAlfrescoToMeterRead :
                       eachAlfrescoId = eachMap.get("alfrescoId")
                       item.listOfIds.append(eachAlfrescoId)
    logger.info('Completed Gettting Alfresco Records for ocrFromECM for '+username+' at '+str(datetime.datetime.now())+" With parameters "+json.dumps(item.listOfIds))         
    if item.listOfIds is None:
        return []
    auth_token = generate_token(url,item.clientId,item.clientSecret,item.oAuthUser,item.oAuthPass)
    logger.info('Completed Gettting Generate Token for ocrFromECM for '+username+' at '+str(datetime.datetime.now())+" With parameters "+json.dumps(auth_token))         
    if auth_token is None : 
        for eachId in item.listOfIds:
            eachResultjson = {"imageId":eachId,"read":"-4"}
            resultJson.append(eachResultjson)
        return resultJson
    listOfImages = []
    for eachId in item.listOfIds:
        base64String=""
        if eachId.startswith("AWSS3"):
           logger.info("Entered AWS function")
           base64String = getbase64EcodedStringfromS3(eachId.split(":")[1])
        else:
            base64String = getbase64EncodedString(auth_token,url,eachId)  # need add the condition
            
        if base64String is not None :
            image = ocrPredict.decodeBitmaptoimage(base64String)
            listOfImages.append(image)
    regions,empty_image = ocrPredict.roi(listOfImages)
    gen_text = ocrPredict.predictBatch(regions,empty_image)
    logger.warn('OCR Completed ocrFromECM for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(gen_text))
    for index,eachGenText in enumerate(gen_text) :   
        if eachGenText is None :
           eachResultjson = {"imageId":item.listOfIds[index],"read":"-1","unit":""}
        else:
            eachResultjson = parseReadResponse(eachGenText,item.listOfIds[index],0)
        # Corelate the OCR and the Manual Read
        if len(mapOfAlfrescoToMeterRead) > 0 :
            for eachmanualRead in mapOfAlfrescoToMeterRead : # list of alfrescoId,manual read, jobId, readType.
                eachAlfrescoId = eachmanualRead["alfrescoId"]
                if eachAlfrescoId == item.listOfIds[index] :
                    eachJsonAppend = {"alfrescoId":eachAlfrescoId,"jobId":eachmanualRead["jobId"],"manualRead":eachmanualRead["meterRead"],"readType":eachmanualRead["readType"]}
                    if item.imageType.lower() == 'retake' and eachmanualRead["mmrMeterRead"] is not None:
                       eachJsonAppend["mmrMeterRead"] =  eachmanualRead["mmrMeterRead"]
                    eachResultjson.update(eachJsonAppend)
                    logger.info('Each ocrFromECM for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(eachResultjson, default=str))
                    break
        resultJson.append(eachResultjson)
    logger.warn('Ended ocrFromECM for '+username+' at '+str(datetime.datetime.now())+" with result "+json.dumps(resultJson,default=str))
    insertedjobs = []
    if len(resultJson) > 0 :
        insertedjobs = insertJobResponse(resultJson,item.imageType)
    return insertedjobs
    
def readFromcsvmultirow (filePath):
    result = {}
    csv.field_size_limit(100000000)
    with open(filePath, newline='') as csvfile:
        idreader = csv.reader(csvfile, delimiter=' ')
        for index,row in enumerate(idreader):
            result[index]=row[0]
        return result
        
def parseReadResponse(gen_text,imageId,totalRecords):
    if gen_text is None :
        return {"imageId":imageId,"read":"-2"}
    else :
        matches = re.findall(r'[A-Za-z ]+|\d+\.\d+|\d+|\.\d+', gen_text)
        if matches is None :
            readSplit[0] = gen_text
        else :
            readSplit = [x.strip() for x in matches]
        meterRead = ""
        if len(readSplit) > 0 :
            meterRead = readSplit[0]
        unit=""
        if len(readSplit) > 1 :
            unit = readSplit[1]
        unitFloat = -1
        try :
            unitFloat = float(unit)
        except ValueError:
            logger.info(" Unit is not a Number")
        if meterRead == 'PF' and unitFloat > 0:
            meterRead = unit
        return {"imageId":imageId,"read":meterRead.lstrip("0"),"unit":unit,"totalRecords":totalRecords}

def readFromcsv (filePath):
    # expect the csv to be a 1 row
    with open(filePath, newline='') as csvfile:
        idreader = csv.reader(csvfile, delimiter=' ')
        for row in idreader:
            return row[0]
        
def getbase64EcodedStringfromS3(key):
    response = s3.get_object(Bucket= S3bucket, Key=key)    
    # Read the content of the image
    image_data = response['Body'].read()
    # Convert the image data to base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    return image_base64

def getbase64EncodedString(auth_token,url, documentId):
    headers = {"Content-Type": "application/json",
               'Authorization': f'Bearer {auth_token}'}
    data = {'id': f'{documentId}'}
    url = url+'/WFMS/getDocDetailsById'
    try:
        response = requests.get(url, params=data, headers=headers)
    except requests.ConnectionError:
        return None
    res_json = response.json()
    try:
        base64String = res_json['data']['docString']
    except ValueError:
        return None
    except TypeError:
        return None
    return base64String
    
def updateJobImage(resultJson,updateBy):
    metadata_obj = MetaData()
    imageTable = Table("mwm_job_images", metadata_obj,Column("id", String, primary_key=True),Column("status", String),Column("ocr_reading",String),Column("ocr_rdg_unit",String),Column("update_by",String),Column("update_date",DateTime),schema="wfms")
    current_dateTime = datetime.datetime.now()
    stmt = (update(imageTable).where(imageTable.c.id == bindparam("identifier")).values({"status":'OCR',"ocr_reading":bindparam("read"),"ocr_rdg_unit":bindparam("unit"),"update_by":updateBy,"update_date":current_dateTime}))
    imageRecords = []
    for eachResult in resultJson :
        imageId = eachResult["imageId"]
        meterRead = eachResult["read"]
        unit = ""
        if "unit" in eachResult:
            unit = eachResult["unit"]
        eachData = {
            "identifier":imageId,
            "read":meterRead.lstrip("0"),
            "unit":unit
        }
        imageRecords.append(eachData)
    if len(imageRecords) > 0 :
        with engine.connect() as conn:
           conn.execute(stmt,imageRecords)
           conn.commit()
           
def insertJobResponse(resultJson,imageType):
    metadata_obj = MetaData()
    jobResponseTable = Table("mwm_job_response", metadata_obj,Column("id", String,primary_key=True, default=uuid4),Column("job_id", String,),Column("attirb_name", String),Column("attribute_value",String),Column("is_material",Integer),Column("created_by",String),Column("create_date",DateTime),schema="wfms")
    current_dateTime = datetime.datetime.now()
    jobList = {}
    for eachRecord in resultJson:
        manualRead = eachRecord["manualRead"]
        ocrRead = eachRecord["read"]
        mmrMeterRead = None
        if imageType.lower() == 'retake' and "mmrMeterRead" in eachRecord:
            mmrMeterRead = eachRecord["mmrMeterRead"]
        isMatched = 'Inconsistent';
        manualReadValue = -1
        ocrReadValue = -1
        if manualRead is None or manualRead == '' :
            manualRead = '-2'
        if ocrRead is None or ocrRead == '' or ocrRead == 'Err' :
            ocrRead = '-2'
        try :
            manualReadValue = Decimal(manualRead)
        except ValueError:
            logger.info(manualRead + " is not a Number")
        except InvalidOperation  :
            logger.info(manualRead + " is not a Number")           
        try :
            ocrReadValue = Decimal(ocrRead)
        except ValueError:
            logger.info(ocrRead + " is not a Number")
        except InvalidOperation  :
            if manualRead is not None : 
                logger.info(str(manualRead) + " is not a Number")
            else : 
                logger.info( " Manual Read is null") 
        if manualReadValue > 0 and ocrReadValue > 0 :
            if manualReadValue == ocrReadValue :
                isMatched = 'Matched'
            else : 
               isMatched = 'Not Matched' 
        jobId = eachRecord["jobId"]
        attributeValue = None
        if jobId in jobList.keys() : 
            attributeValue = jobList[jobId]
        readType = eachRecord["readType"]
        alfElementName = readType+"alfrescoId"
        newAttributeValue = {"OCR Result":isMatched,"Reading in DB":manualRead,"Reading from OCR":ocrRead,"unit":eachRecord["unit"],"readType":readType,alfElementName:eachRecord["alfrescoId"]}
        if mmrMeterRead is not None :
            newAttributeValue["mmrMeterRead"] = mmrMeterRead
        actualAttribValue = []
        if attributeValue is not None : 
            actualAttribValue = json.loads(attributeValue)
        actualAttribValue.append(newAttributeValue)
        jobList[jobId] = json.dumps(actualAttribValue, default=str)
    logger.info('Attempting to Insert ocrFromECM at '+str(datetime.datetime.now())+" with result "+json.dumps(jobList, default=str))
    eachData = []
    for eachJob in jobList:
        attribValue = jobList[eachJob].strip()
        attribValue = attribValue.replace('\\', '') 
        eachDataStr = {
            "jobId":eachJob,
            "attrib_value":attribValue
        }
        eachData.append(eachDataStr)       
    logger.info('Attempting to Insert ocrFromECM finally at '+str(datetime.datetime.now())+" with result "+json.dumps(eachData, default=str))
    attribName='New Meter OCR Result'
    if imageType.lower() == 'old':
        attribName = 'Old Meter OCR Result'
    elif imageType.lower() == 'retake':
        attribName = 'Old Meter OCR Retake'
    stmt = (insert(jobResponseTable).values({"job_id":bindparam("jobId"),"attirb_name":attribName,"attribute_value":bindparam("attrib_value"),"created_by":"python","create_date":current_dateTime,"is_material":0}))
    with engine.connect() as conn:
        conn.execute(stmt,eachData)
        conn.commit()
    return eachData