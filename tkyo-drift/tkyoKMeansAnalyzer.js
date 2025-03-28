import { kmeans } from 'ml-kmeans';
import path from 'path'
import fs from 'fs'


 const readVectorsFromBin = async () => {

    // Load the raw binary blob
    const stream = fs.createReadStream(path.join('/home/msymeono/Codesmith/projects/tkyo-drift', 'semantic_input.training.bin'), {
        highWaterMark:1536
      })

    const vectorList = [];

    // Convert the blob into a Float32 Array
    for await (const chunk of stream) {
      // Guard against partial chunks
      if (chunk.length !== 384* 4) continue;

      const floatArray = new Float32Array(
        chunk.buffer,
        chunk.byteOffset,
        384
      );
      vectorList.push(floatArray);
    }

    // Determine if we have less vectors than the rolling max size
    const totalVectors = vectorList.length;
    const vectorCount = Math.min(50000, totalVectors);

    // Calculate how many vectors we need to pull out of the float array
    const startIndex = 0;

    // Set vector array to an array of arrays equal to the size of vector count
    const vectorArray = new Array(vectorCount);

    // For each dim length, push the numbers into a vector array
    for (let i = 0; i < vectorCount; i++) {
      // Calculate the start and end positions need to pull out of from the float array
      const vector = vectorList[startIndex + i];

      // Assign the vector from the float array to the vector array
      vectorArray[i] = vector;
    }
    return vectorArray
  }


const kMeans = async () => {
console.time('reading vectors from bin')
const vectors = await readVectorsFromBin()
console.timeEnd('reading vectors from bin')
const testK = 158
const testK2 = 329
const testK3 = 500
const testK4 = 750
const testK5 = 1000

    const ks = [testK, testK2, testK3, testK4, testK5];
    const inertias = [];

    const calculateInertia = (vectors, result) => {
        let inertia = 0;
        for (let i = 0; i < vectors.length; i++) {
            const clusterIndex = result.clusters[i];
            const centroid = result.centroids[clusterIndex];
            //calculate the euclideanDistance
            const euclideanDistance = Math.sqrt(
                vectors[i].reduce((sum, val, idx) => sum + (val - centroid[idx]) ** 2, 0)
            );
            // inertia is equal to the sum of each Euclidean Distance squared, so we square each eD and add it to our inertia
            inertia += euclideanDistance ** 2;
          
        }
        return inertia;
    };

    for (let k of ks) {
        console.time('kmeans for loop')
        const result = kmeans(vectors, k, { initialization: 'kmeans++' });
        // console.log('result:',result)
        const inertia = calculateInertia(vectors, result);
        
        console.log('ratio', k, inertia)
        inertias.push(inertia);
        console.timeEnd('kmeans for loop')
    }
    
    // console.log("Ks:", ks);
    // console.log("Inertias:", inertias);
};

console.time('kmeans')
await kMeans();
console.timeEnd('kmeans')


// inertia
// 4.79586560776999 => 19,183
// ratio 4.157074914839581 => 18,706
// ratio 3.6377612481786805 => 18188.8062408934025