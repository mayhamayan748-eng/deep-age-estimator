/**
 * Copies a picked or dropped file into memory right away.
 *
 * On Android, the browser's permission to read a photo chosen from the gallery or Google Photos can be
 * revoked once the <input> is reset or the page moves on, and later reads fail with NotReadableError.
 * Reading the bytes immediately keeps a usable copy. Returns the original file if the read fails, so the
 * caller can report the error.
 */
export async function readPickedFile(file: File): Promise<File> {
  try {
    const bytes = await file.arrayBuffer();
    return new File([bytes], file.name, { type: file.type, lastModified: file.lastModified });
  } catch {
    return file;
  }
}
